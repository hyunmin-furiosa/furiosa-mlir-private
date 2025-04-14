#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/bit.h"

namespace mlir::furiosa::sfr {

template <typename DT> inline DT get_bit_mask(unsigned int size) {
  using DT_U = std::make_unsigned_t<DT>;
  DT_U mask = ~(DT_U)0;
  if (size >= 8 * sizeof(DT)) {
    return llvm::bit_cast<DT>(mask);
  } else {
    mask = ~(mask << size);
    return llvm::bit_cast<DT>(mask);
  }
}

template <typename DT>
inline DT insert_bits(const DT &v, const DT &rhs, unsigned int size,
                      unsigned int offset) {
  DT mask = get_bit_mask<DT>(size);

  DT new_data = (rhs & mask);
  mask <<= offset;
  new_data <<= offset;

  return (v & ~mask) | new_data;
}

template <typename DT>
inline DT extract_bits(const DT &v, unsigned int size, unsigned int offset) {
  const DT mask = get_bit_mask<DT>(size);
  return (v >> offset) & mask;
}

template <class DT>
inline DT partial_value_sign_extension(const DT &value, unsigned int offset) {
  if ((value >> (offset - 1)) & 0x1) {
    return value | (~((1ull << offset) - 1));
  } else
    return value;
}

template <class DT> class Register;
template <class DT> class Bitfield;
template <class DT> class BitfieldContainer;

// storage
template <class DT> class Block {
public:
  using data_type = DT;
  using this_type = Block<DT>;
  using register_wrapper = std::reference_wrapper<sfr::Register<DT>>;
  explicit Block(unsigned int base, unsigned int size, bool as_memory = false)
      : base_(base), size_(size), as_memory_(as_memory),
        block_(size / sizeof(DT)) {}
  DT &get(const std::uint32_t &offset) { return block_[offset / sizeof(DT)]; }
  void put(const std::uint32_t &offset, const DT &data) {
    const bool valid = is_valid(offset);
    const bool reg_valid = valid & (valid_map_.count(offset) != 0);
    const bool mem_valid = valid & as_memory_;
    if (reg_valid) {
      child_map_.at(offset).get().put(data);
    } else if (mem_valid) {
      block_[offset / sizeof(DT)] = data;
    }
  }
  void reset() {
    for (auto &[offset, reg] : child_map_) {
      reg.get().reset();
    }
  }
  void register_child(Register<DT> &child, std::uint32_t offset) {
    if (valid_map_.count(offset) == 0) {
      valid_map_[offset] = true;
      child_map_.emplace(offset, std::ref(child));
    } else {
      // There are multiple reigster at the same offset
    }
  }

  DT &operator[](unsigned int index) {
    assert(index < block_.size());
    return block_[index];
  }

  DT operator[](unsigned int index) const {
    assert(index < block_.size());
    return block_[index];
  }
  void *ptr(const std::uint32_t offset) {
    auto base = reinterpret_cast<std::uint8_t *>(block_.data());
    return reinterpret_cast<void *>(base + offset);
  }

  std::uint32_t get_base() { return base_; }
  std::uint32_t get_size() { return size_; }
  std::vector<DT> &get_blocks() { return block_; }

  // offset address, not index
  bool is_valid(const std::uint32_t &offset) {
    return as_memory_ ? (offset <= (size_ - sizeof(DT)))
                      : (valid_map_.count(offset) != 0);
  }

private:
  unsigned int base_;
  unsigned int size_;
  bool as_memory_;
  // offset address and valid
  std::map<std::uint32_t, bool> valid_map_{};
  std::map<std::uint32_t, register_wrapper> child_map_{};
  std::vector<DT> block_;
};

template <class DT> class Register {
public:
  using data_type = DT;

public:
  Register(const std::string &name, Block<DT> &parent,
           const std::uint32_t offset, const DT reset = 0x0,
           const DT read_only = 0x0)
      : block_{parent}, register_{parent.get(offset)}, offset_{offset},
        reset_{reset}, read_only_{read_only} {
    parent.register_child(*this, offset);
    put(reset_, true);
  }
  Register(const Register &r) = default;
  ~Register() {}

  void set_read_only(const DT &mask) { read_only_ = mask; }
  void put(const DT &data, bool force = false) {
    if (force || read_only_ == 0x0) {
      register_ = data;
    } else {
      register_ = (~read_only_ & data) | (read_only_ & register_);
    }
  }

  DT get() const { return register_; }
  DT &get() { return register_; }

  void reset() { put(reset_, true); }

  void register_child(Bitfield<DT> *child, const std::uint32_t &size,
                      const std::uint32_t &offset) {
    // TODO
  }

  operator DT() const { return get(); }

  Register &operator=(DT value) {
    put(value);
    return *this;
  }
  Register &operator=(const Register &r) {
    // TODO
    return *this;
  }

  Register &operator+=(DT value) {
    register_ += value;
    return *this;
  }
  Register &operator-=(DT value) {
    register_ -= value;
    return *this;
  }
  Register &operator/=(DT value) {
    register_ /= value;
    return *this;
  }
  Register &operator*=(DT value) {
    register_ *= value;
    return *this;
  }
  Register &operator%=(DT value) {
    register_ %= value;
    return *this;
  }
  Register &operator^=(DT value) {
    register_ ^= value;
    return *this;
  }
  Register &operator&=(DT value) {
    register_ &= value;
    return *this;
  }
  Register &operator|=(DT value) {
    register_ |= value;
    return *this;
  }
  Register &operator>>=(DT value) {
    register_ >>= value % sizeof(DT);
    return *this;
  }
  Register &operator<<=(DT value) {
    register_ <<= value % sizeof(DT);
    return *this;
  }

  Register &operator--() {
    register_--;
    return *this;
  }
  DT operator--(int) { return register_--; }
  Register &operator++() {
    register_++;
    return *this;
  }
  DT operator++(int) { return register_++; }

private:
  Block<DT> &block_;
  DT &register_;
  std::uint32_t offset_;
  DT reset_;
  DT read_only_;

  BitfieldContainer<DT> bitfield_container_;
};

template <class DT> class Bitfield {
public:
  using data_type = DT;

public:
  Bitfield(const std::string &name, Register<DT> &parent, unsigned int size,
           unsigned int offset)
      : register_{parent.get()}, size_{size}, offset_{offset} {
    parent.register_child(this, size_, offset_);
  }
  Bitfield(const Bitfield &b) = default;
  ~Bitfield() {}

  void put(const DT &value) {
    DT data = insert_bits(register_, value, size_, offset_);
    register_ = data;
  }
  void put(const DT &value, const DT &bit_mask) {
    const DT masked_data =
        (extract_bits(register_, size_, offset_) & ~bit_mask) |
        (value & bit_mask);
    DT data = insert_bits(register_, masked_data, size_, offset_);
    register_ = data;
  }
  DT get() const { return extract_bits(register_, size_, offset_); }
  DT get(const DT &bit_mask) const {
    return extract_bits(register_, size_, offset_) & bit_mask;
  }
  operator DT() const { return get(); }

  Bitfield &operator=(DT value) {
    put(value);
    return *this;
  }
  Bitfield &operator=(const Bitfield &b) {
    put(b);
    return *this;
  }

  Bitfield &operator+=(DT value) {
    put(get() + value);
    return *this;
  }
  Bitfield &operator-=(DT value) {
    put(get() - value);
    return *this;
  }
  Bitfield &operator/=(DT value) {
    put(get() / value);
    return *this;
  }
  Bitfield &operator*=(DT value) {
    put(get() * value);
    return *this;
  }

  Bitfield &operator%=(DT value) {
    put(get() % value);
    return *this;
  }
  Bitfield &operator^=(DT value) {
    put(get() ^ value);
    return *this;
  }
  Bitfield &operator&=(DT value) {
    put(get() & value);
    return *this;
  }
  Bitfield &operator|=(DT value) {
    put(get() | value);
    return *this;
  }
  Bitfield &operator<<=(DT value) {
    put(get() << value);
    return *this;
  }

  Bitfield &operator>>=(DT value) {
    put(get() >> value);
    return *this;
  }

  Bitfield &operator--() {
    put(get() - 1);
    return *this;
  }
  DT operator--(int) {
    DT tmp = *this;
    --(*this);
    return tmp;
  }
  Bitfield &operator++() {
    put(get() + 1);
    return *this;
  }
  DT operator++(int) {
    DT tmp = *this;
    ++(*this);
    return tmp;
  }
  template <class T> T to() {
    if constexpr (std::is_signed_v<T>) {
      return static_cast<T>(partial_value_sign_extension(get(), size_));
    } else {
      return static_cast<T>(get());
    }
  }

private:
  // Register<DT> &register_;
  DT &register_;
  unsigned int size_, offset_;
};

template <class DT> class BitfieldContainer {
public:
  BitfieldContainer() {}

private:
  std::vector<Bitfield<DT> *> bitfields_;
};

} // namespace mlir::furiosa::sfr
