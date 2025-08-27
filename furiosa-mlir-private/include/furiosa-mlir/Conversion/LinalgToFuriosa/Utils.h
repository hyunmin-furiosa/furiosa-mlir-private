#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir::furiosa {

static constexpr auto SLICE_SRAM_SIZE = 0x400000; // 4MB
static constexpr auto DRAM_ACCESS_WIDTH = 256;
using indexer_t =
    std::pair<SmallVector<std::int64_t>, SmallVector<std::int64_t>>;
using indexers_t = std::pair<indexer_t, indexer_t>;
using ValueMapper = llvm::DenseMap<Value, std::int64_t>;

std::uint64_t getSimpleMultiplier(AffineMap map) {
  auto exprs = map.getResults();
  assert(exprs.size() == 1);
  return exprs[0].getLargestKnownDivisor();
}

indexer_t getPartitioningIndexer(llvm::SmallVector<OpFoldResult> offsets,
                                 ValueMapper value_mapper) {
  llvm::SmallVector<std::int64_t> partitioning_limits;
  llvm::SmallVector<std::int64_t> partitioning_strides;
  for (auto offset : offsets) {
    if (llvm::isa<Value>(offset)) {
      auto value = llvm::dyn_cast_or_null<Value>(offset);
      if (auto defining_op = value.getDefiningOp()) {
        auto apply_op =
            llvm::dyn_cast_or_null<affine::AffineApplyOp>(defining_op);
        assert(apply_op.getMapOperands().size() == 1);
        auto operand = apply_op.getMapOperands()[0];
        partitioning_limits.push_back(value_mapper[operand]);
        partitioning_strides.push_back(getSimpleMultiplier(apply_op.getMap()));
      } else {
        partitioning_limits.push_back(value_mapper[value]);
        partitioning_strides.push_back(1);
      }
    } else {
      auto dim = llvm::dyn_cast_or_null<mlir::IntegerAttr>(
                     llvm::dyn_cast_or_null<mlir::Attribute>(offset))
                     .getInt();
      partitioning_limits.push_back(1);
      partitioning_strides.push_back(dim);
    }
  }

  return {partitioning_limits, partitioning_strides};
}

indexers_t optimizeIndexers(indexers_t indexers) {
  auto [source_indexer, destination_indexer] = indexers;
  SmallVector<indexer_t> indexers_vector = {source_indexer,
                                            destination_indexer};
  SmallVector<std::uint32_t> consecutive_dims;
  SmallVector<std::uint32_t> consecutive_sizes;
  for (auto indexer : indexers_vector) {
    auto [limits, strides] = indexer;
    auto rank = limits.size();
    for (auto i = 1u; i < rank; i++) {
      auto inner_size = limits[i - 1] * strides[i - 1];
      if (strides[i] != inner_size) {
        consecutive_dims.push_back(i - 1);
        consecutive_sizes.push_back(inner_size);
        break;
      }
    }
  }

  assert(consecutive_dims[0] == consecutive_dims[1]);
  assert(consecutive_sizes[0] == consecutive_sizes[1]);
  auto max_consecutive_dim =
      *std::max_element(consecutive_dims.begin(), consecutive_dims.end());
  auto max_consecutive_size =
      *std::max_element(consecutive_sizes.begin(), consecutive_sizes.end());

  for (auto &indexer : indexers_vector) {
    auto &[limits, strides] = indexer;
    limits.erase(limits.begin(), limits.begin() + max_consecutive_dim);
    strides.erase(strides.begin(), strides.begin() + max_consecutive_dim);
    if ((max_consecutive_size >= DRAM_ACCESS_WIDTH) &&
        (max_consecutive_size % DRAM_ACCESS_WIDTH == 0)) {
      limits.insert(limits.begin(), max_consecutive_size / DRAM_ACCESS_WIDTH);
      strides.insert(strides.begin(), DRAM_ACCESS_WIDTH);
    } else {
      limits.insert(limits.begin(), max_consecutive_size);
      strides.insert(strides.begin(), 1);
    }
  }

  return std::make_pair(indexers_vector[0], indexers_vector[1]);
}

indexers_t getIndexers(RankedTensorType dram_type, RankedTensorType sram_type,
                       llvm::SmallVector<OpFoldResult> offsets,
                       ValueMapper value_mapper) {
  assert(dram_type.getRank() == sram_type.getRank());
  auto rank = dram_type.getRank();
  auto dram_shape = dram_type.getShape();
  auto sram_shape = sram_type.getShape();
  auto slice_partition_shape = SmallVector<std::int64_t>(rank);
  auto limits = SmallVector<std::int64_t>();
  auto dram_strides = SmallVector<std::int64_t>();
  auto sram_strides = SmallVector<std::int64_t>();

  // find in-slice shape
  auto dram_stride = 1;
  auto sram_stride = 1;
  for (auto i = rank - 1; i >= 0; i--) {
    auto dram_limit = dram_shape[i];
    auto sram_limit = sram_shape[i];
    slice_partition_shape[i] = dram_limit / sram_limit;

    limits.push_back(sram_limit);
    dram_strides.push_back(dram_stride);
    sram_strides.push_back(sram_stride);

    dram_stride *= dram_limit;
    sram_stride *= sram_limit;
  }

  // find slice partitioning shape
  auto [partitioning_limits, partitioning_strides] =
      getPartitioningIndexer(offsets, value_mapper);

  dram_stride = 1;
  sram_stride = 1;
  for (auto i = rank - 1; i >= 0; i--) {
    auto dram_limit = dram_shape[i];
    auto partitioning_limit = partitioning_limits[i];
    auto partitioning_stride = partitioning_strides[i];

    limits.push_back(partitioning_limit);
    dram_strides.push_back(dram_stride * partitioning_stride);
    sram_strides.push_back(sram_stride * SLICE_SRAM_SIZE);

    dram_stride *= dram_limit;
    sram_stride *= partitioning_limit;
  }

  // broadcast remaining slice partitions
  static constexpr auto num_slices = 64;
  auto broadcast_limit = num_slices;
  for (auto limit : partitioning_limits) {
    broadcast_limit /= limit;
  }
  if (broadcast_limit > 1) {
    limits.push_back(broadcast_limit);
    dram_strides.push_back(0);
    sram_strides.push_back(0);
  }

  indexer_t dram_indexer = {limits, dram_strides};
  indexer_t sram_indexer = {limits, sram_strides};
  return optimizeIndexers({dram_indexer, sram_indexer});
}

} // namespace mlir::furiosa
