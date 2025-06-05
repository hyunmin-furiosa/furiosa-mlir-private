#pragma once

#include "furiosa-mlir/Dialect/Furiosa/IR/FuriosaOps.h"

namespace mlir::furiosa {

static constexpr std::uint64_t DRAM_RESERVED_SIZE =
    0x400000; // 4MB reserved for program binary

class Allocator {
public:
  Allocator() = default;

  std::uint64_t allocate(size_t size) {
    allocated_addresses.insert(current_address);
    auto old_address = current_address;
    current_address += size;
    return old_address;
  }

  void deallocate(std::uint64_t address) { allocated_addresses.erase(address); }

private:
  std::uint64_t current_address;
  std::set<std::uint64_t> allocated_addresses;
};

class MemoryAllocator {
public:
  MemoryAllocator() {
    auto num_allocators = getMaxEnumValForMemoryType();
    for (auto i = 0u; i < num_allocators; ++i) {
      allocators[static_cast<MemoryType>(i)] = Allocator();
    }
    allocators[MemoryType::dram].allocate(DRAM_RESERVED_SIZE);
  };

  std::uint64_t allocate(size_t size, MemoryType memory_type) {
    auto &allocator = allocators[memory_type];
    return allocator.allocate(size);
  }

  void deallocate(std::uint64_t address, MemoryType memory_type) {
    auto &allocator = allocators[memory_type];
    allocator.deallocate(address);
  }

private:
  std::map<MemoryType, Allocator> allocators;
};

} // namespace mlir::furiosa
