#include <stdint.h>
#include <stddef.h>

void *memcpy(void *dest, const void *src, size_t num) {
  if ((uintptr_t)dest % sizeof(uint64_t) == 0 &&
      (uintptr_t)src % sizeof(uint64_t) == 0 && num % sizeof(uint64_t) == 0) {
    uint64_t *destination = (uint64_t *)dest;
    const uint64_t *source = (uint64_t *)src;
    for (size_t count = 0; count < num / sizeof(uint64_t); count++) {
      *destination++ = *source++;
    }
  } else {
    char *destination = (char *)dest;
    const char *source = (char *)src;
    for (size_t count = 0; count < num; count++) {
      *destination++ = *source++;
    }
  }
  return dest;
}
