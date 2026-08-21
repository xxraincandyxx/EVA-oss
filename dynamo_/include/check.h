// Copyright [Eternal] xxraincandyxx. All rights reserved.
//
// A simplified, self-contained implementation of glog's CHECK/DCHECK macros,
// adhering to the Google C++ Style Guide. This header provides utilities for
// enforcing invariants in code; they will terminate the program if a condition
// is not met.

#ifndef JIEHEADERGUARD_CHECK_
#define JIEHEADERGUARD_CHECK_
// NOLINTBEGIN

#include <cstdlib>   // For std::abort
#include <iostream>  // For std::cerr
#include <memory>    // For std::unique_ptr
#include <sstream>   // For std::ostringstream

// LIKELY/UNLIKELY: Provides hints to the compiler for branch prediction.
// CHECKs are expected to pass, so the condition is LIKELY(true).
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

namespace my_glog::internal {

// This class is used to implement the fatal logging stream.
// It is constructed when a CHECK fails. It captures the failure message and,
// upon destruction, prints the message to stderr and aborts the program.
// This leverages the RAII (Resource Acquisition Is Initialization) pattern.
class CheckHelper {
 public:
  CheckHelper(const char* file, int line, const char* condition_str);

  // The destructor is the key part of this class. It is a [[noreturn]] function,
  // meaning it will not return to the caller. It logs the complete message
  // and terminates the application.
  [[noreturn]] ~CheckHelper();

  // Returns the stream to which the user can append a custom message.
  // e.g., CHECK(x > 0) << "x must be positive, but was " << x;
  std::ostream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
};

CheckHelper::CheckHelper(const char* file, int line,
                         const char* condition_str) {
  stream_ << "Check failed at " << file << ":" << line << ": " << condition_str
          << "\n";
}

CheckHelper::~CheckHelper() {
  std::cerr << stream_.str() << std::endl;
  std::abort();
}

// A helper class for implementing the binary CHECKs (e.g., CHECK_EQ).
// It captures the arguments and their string representations to provide a more
// informative error message (e.g., "Check failed: a == b (5 vs. 6)").
template <typename T1, typename T2>
class CheckOpHelper {
 public:
  CheckOpHelper(const char* file, int line, const T1& val1,
                const char* val1_str, const T2& val2, const char* val2_str,
                const char* op_str)
      : file_(file),
        line_(line),
        val1_str_(val1_str),
        val2_str_(val2_str),
        op_str_(op_str),
        val1_(&val1),
        val2_(&val2) {}

  // This is the "sink" that is returned when the check succeeds. It's a
  // temporary object that does nothing, effectively eating the streamed message.
  // The templated operator<< allows it to accept any type.
  struct NullStream {
    template <typename T>
    NullStream& operator<<(const T&) {
      return *this;
    }
  };

  // If the check fails, this function returns a unique_ptr to a CheckHelper,
  // which will then be used to stream the user's message and abort.
  // If the check succeeds, it returns nullptr.
  std::unique_ptr<CheckHelper> MakeCheckHelper() const {
    auto helper = std::make_unique<CheckHelper>(file_, line_, "");
    helper->stream() << "Check failed: " << val1_str_ << " " << op_str_ << " "
                     << val2_str_ << " (" << *val1_ << " vs. " << *val2_ << ")";
    return helper;
  }

 private:
  const char* file_;
  const int line_;
  const char* val1_str_;
  const char* val2_str_;
  const char* op_str_;
  const T1* val1_;
  const T2* val2_;
};

}  // namespace my_glog::internal

// The core CHECK macro.
//
// How it works:
// 1. The condition is evaluated.
// 2. We use a standard `if-else` statement to control the flow. This ensures
//    the macro behaves like a single statement (e.g., `if (foo) CHECK(bar);`).
// 3. If the condition is `true` (the LIKELY case), the `if` branch is taken,
//    which does nothing.
// 4. If the condition is `false`, the `else` branch is taken. It creates a
//    temporary `CheckHelper` object on the stack. The `stream()` method is
//    called, and the user's message (if any) is streamed into it.
// 5. At the end of the full statement (at the semicolon), the temporary
//    `CheckHelper` object is destroyed. Its destructor is called, which prints
//    the fatal message and calls `std::abort()`.
#define CHECK(condition) \
  if (LIKELY(condition)) \
    ;                    \
  else                   \
    ::my_glog::internal::CheckHelper(__FILE__, __LINE__, #condition).stream()

// The debug-only version of CHECK.
// In debug builds (when NDEBUG is not defined), it expands to a normal CHECK.
// In release builds, it expands to a `while(false)` loop, which is a common
// C++ trick to create a statement that does nothing and safely "eats" any
// streamed arguments `<< ...;` without evaluating its condition.
#if defined(NDEBUG) && !defined(DCHECK_ALWAYS_ON)
#define DCHECK(condition) \
  while (false)           \
  std::cout
#else
#define DCHECK(condition) CHECK(condition)
#endif

// Helper macro for binary comparisons (CHECK_EQ, CHECK_NE, etc.).
//
// How it works:
// 1. It creates a CheckOpHelper to store the values and their string forms.
// 2. It calls the comparison function (`op_func`) on the values.
// 3. If the comparison fails, `MakeCheckHelper()` creates a real logger.
// 4. If it succeeds, `MakeCheckHelper()` would return nullptr, but we use a
//    ternary operator to return a do-nothing NullStream sink instead.
#define CHECK_OP(op_func, op_str, val1, val2)                    \
  if (auto helper = ::my_glog::internal::CheckOpHelper(          \
          __FILE__, __LINE__, val1, #val1, val2, #val2, op_str); \
      !op_func(val1, val2))                                      \
  helper.MakeCheckHelper()->stream()

// Define standard comparison operators.
namespace my_glog::internal {

template <typename T1, typename T2>
inline bool IsEqual(const T1& a, const T2& b) {
  return a == b;
}

template <typename T1, typename T2>
inline bool IsNotEqual(const T1& a, const T2& b) {
  return a != b;
}

template <typename T1, typename T2>
inline bool IsLess(const T1& a, const T2& b) {
  return a < b;
}

template <typename T1, typename T2>
inline bool IsLessEqual(const T1& a, const T2& b) {
  return a <= b;
}

template <typename T1, typename T2>
inline bool IsGreater(const T1& a, const T2& b) {
  return a > b;
}

template <typename T1, typename T2>
inline bool IsGreaterEqual(const T1& a, const T2& b) {
  return a >= b;
}

}  // namespace my_glog::internal

// Public-facing binary comparison macros.
#define CHECK_EQ(val1, val2) \
  CHECK_OP(::my_glog::internal::IsEqual<>, "==", val1, val2)
#define CHECK_NE(val1, val2) \
  CHECK_OP(::my_glog::internal::IsNotEqual<>, "!=", val1, val2)
#define CHECK_LT(val1, val2) \
  CHECK_OP(::my_glog::internal::IsLess<>, "<", val1, val2)
#define CHECK_LE(val1, val2) \
  CHECK_OP(::my_glog::internal::IsLessEqual<>, "<=", val1, val2)
#define CHECK_GT(val1, val2) \
  CHECK_OP(::my_glog::internal::IsGreater<>, ">", val1, val2)
#define CHECK_GE(val1, val2) \
  CHECK_OP(::my_glog::internal::IsGreaterEqual<>, ">=", val1, val2)

// Debug-only versions of the binary comparison macros.
#if defined(NDEBUG) && !defined(DCHECK_ALWAYS_ON)
#define DCHECK_EQ(val1, val2) \
  while (false)               \
  std::cout
#define DCHECK_NE(val1, val2) \
  while (false)               \
  std::cout
#define DCHECK_LT(val1, val2) \
  while (false)               \
  std::cout
#define DCHECK_LE(val1, val2) \
  while (false)               \
  std::cout
#define DCHECK_GT(val1, val2) \
  while (false)               \
  std::cout
#define DCHECK_GE(val1, val2) \
  while (false)               \
  std::cout
#else
#define DCHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define DCHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define DCHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define DCHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define DCHECK_GT(val1, val2) CHECK_GT(val1, val2)
#define DCHECK_GE(val1, val2) CHECK_GE(val1, val2)
#endif  // NDEBUG

// NOLINTEND
#endif  // JIEHEADERGUARD_CHECK_
