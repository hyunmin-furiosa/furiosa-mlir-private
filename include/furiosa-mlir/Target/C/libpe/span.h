#ifndef _SPAN_H
#define _SPAN_H

// Should match with src/span.rs#CSpan
struct span_handle_t {
  const char *name;
  uint64_t profile_uid;
  uint64_t begin_cycle;
};
static_assert(sizeof(struct span_handle_t) == 24);

#endif // _SPAN_H
