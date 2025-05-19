#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir::furiosa {

static constexpr auto SLICE_SRAM_SIZE = 0x400000; // 4MB
using indexer_t =
    std::pair<SmallVector<std::int64_t>, SmallVector<std::int64_t>>;
using indexers_t = std::pair<indexer_t, indexer_t>;
using lower_upper_step_t = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
using ValueMapper = llvm::DenseMap<Value, lower_upper_step_t>;

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
        auto [lower_bound, upper_bound, step] = value_mapper[operand];
        partitioning_limits.push_back((upper_bound - lower_bound) / step);
        partitioning_strides.push_back(getSimpleMultiplier(apply_op.getMap()));
      } else {
        auto [lower_bound, upper_bound, step] = value_mapper[value];
        partitioning_limits.push_back((upper_bound - lower_bound) / step);
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
    auto sram_limit = sram_shape[i];
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
  return {dram_indexer, sram_indexer};
}

} // namespace mlir::furiosa
