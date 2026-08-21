// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ xxraincandyxx/JIE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Module:        jie::log (FINAL, ROBUST, AND CORRECTLY STRINGIFYING VERSION)
// Documentation: https://github.com/DmitriBogdanov/UTL/blob/master/docs/module_log.md
// Source repo:   https://github.com/xxraincandyxx/EVA/dynamo_/include
//
// This project is forked and highly-refactored, and is licensed under the MIT License
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#if !defined(JIE_PICK_MODULES) || defined(JIEMODULE_LOG)
#ifndef JIEHEADERGUARD_LOG_
#define JIEHEADERGUARD_LOG_

// _______________________ INCLUDES _______________________
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>  // std::put_time
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

// ____________________ IMPLEMENTATION ____________________
namespace jie::log {

// ===================================================================
// --- Core Types and Enums (Public API) ---
// ===================================================================

/**
 * @enum Verbosity
 * @brief Defines the severity level of a log message.
 *
 * Sinks can be configured to only log messages up to a certain verbosity level.
 * A sink with level `INFO` will log `ERR`, `WARN`, `NOTE`, and `INFO`, but not `DEBUG` or `TRACE`.
 */
enum class Verbosity : int8_t {
  ERR = 0,  ///< Critical errors that might prevent the program from continuing.
  WARN = 1,   ///< Warnings about potential issues that don't stop execution.
  NOTE = 2,   ///< Noteworthy events, more important than INFO.
  INFO = 3,   ///< General informational messages about program state.
  DEBUG = 4,  ///< Detailed information for debugging purposes.
  TRACE = 5   ///< Highly detailed execution tracing, for deep debugging.
};

/**
 * @enum OpenMode
 * @brief Specifies how to open a log file.
 */
enum class OpenMode : int8_t {
  REWRITE,  ///< Overwrite the file if it exists.
  APPEND    ///< Append to the file if it exists.
};

/**
 * @enum Colors
 * @brief Specifies whether to use ANSI color codes in the output.
 * @note This typically only has a visible effect on `ostream` sinks connected to a compatible terminal.
 */
enum class Colors : int8_t {
  ENABLE,  ///< Enable ANSI color codes.
  DISABLE  ///< Disable ANSI color codes.
};

/**
 * @struct Columns
 * @brief A set of flags to control which components are included in the formatted log message.
 *
 * This allows for fine-grained control over the log output format for each sink.
 * For example, one file can have full details, while another might only log the raw message.
 */
struct Columns {
  bool datetime =
      true;  ///< [YYYY-MM-DD HH:MM:SS.ms] The calendar time of the log event.
  bool uptime =
      true;  ///< [ 0000.123s] The time elapsed since the logger was first used.
  bool thread =
      true;  ///< [thread:12345] The ID of the thread that generated the log.
  bool level = true;  ///< [INFO] The verbosity level of the message.
  bool callsite =
      true;  ///< [file:line function()] The source location of the log call.
  bool message = true;  ///< The user-provided log message itself.
};

/// @brief Helper macro to get the function name in a cross-compiler way.
#if defined(__GNUC__) || defined(__clang__)
#define JIE_LOG_GET_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define JIE_LOG_GET_FUNCTION_NAME __FUNCSIG__
#else
#define JIE_LOG_GET_FUNCTION_NAME __func__
#endif

/**
 * @struct Callsite
 * @brief Captures the source location where a log message was generated.
 *
 * This struct is automatically populated by the `JIE_LOG_*` macros.
 */
struct Callsite {
  std::string_view file;
  int line;
  std::string_view function;
};

// ===================================================================
// --- THE ORIGINAL, POWERFUL STRINGIFIER (RESTORED) ---
// ===================================================================
namespace internal {
#define jie_log_define_trait(trait_name_, ...)                                 \
  template <class T, class = void>                                             \
  struct trait_name_ : std::false_type {};                                     \
  template <class T>                                                           \
  struct trait_name_<T, std::void_t<decltype(__VA_ARGS__)>> : std::true_type { \
  };                                                                           \
  template <class T>                                                           \
  constexpr bool trait_name_##_v = trait_name_<T>::value

jie_log_define_trait(_has_begin, std::declval<T>().begin());
jie_log_define_trait(_has_end, std::declval<T>().end());
jie_log_define_trait(_has_input_it, std::next(std::declval<T>().begin()));
jie_log_define_trait(_has_get, std::get<0>(std::declval<T>()));
jie_log_define_trait(_has_tuple_size, std::tuple_size<T>::value);
jie_log_define_trait(_has_ostream_insert, std::declval<std::ostream>()
                                              << std::declval<T>());

#undef jie_log_define_trait

template <class>
constexpr bool _always_false_v = false;

struct StringifierBase { /* ... Unchanged ... */
  using self = StringifierBase;

  template <class T>
  static void append(std::string& buffer, const T& value) {
    _append_selector(buffer, value);
  }

  template <class... Args>
  static void append(std::string& buffer, const Args&... args) {
    (self::append(buffer, args), ...);
  }

  template <class... Args>
  [[nodiscard]] static std::string stringify(Args&&... args) {
    std::string buffer;
    self::append(buffer, std::forward<Args>(args)...);
    return buffer;
  }

 private:
  template <class T>
  static void _append_selector(std::string& buffer, const T& value) {
    using DecayedT = std::decay_t<T>;
    if constexpr (std::is_same_v<DecayedT, bool>) {
      buffer += value ? "true" : "false";
    } else if constexpr (std::is_same_v<DecayedT, char>) {
      buffer += value;
    } else if constexpr (std::is_convertible_v<DecayedT, std::string_view>) {
      buffer += std::string_view(value);
    } else if constexpr (std::is_integral_v<DecayedT>) {
      char str_buffer[21];
      auto res = std::to_chars(str_buffer, str_buffer + 21, value);
      buffer.append(str_buffer, res.ptr - str_buffer);
    } else if constexpr (std::is_floating_point_v<DecayedT>) {
      char str_buffer[32];
      auto ret = std::snprintf(str_buffer, sizeof(str_buffer), "%.6g",
                               static_cast<double>(value));
      if (ret > 0 && ret < static_cast<int>(sizeof(str_buffer))) {
        buffer.append(str_buffer, ret);
      } else {
        buffer.append("[fp_err]", 8);
      }
    } else if constexpr (std::is_enum_v<DecayedT>) {
      _append_selector(buffer,
                       static_cast<std::underlying_type_t<DecayedT>>(value));
    } else if constexpr (_has_begin_v<DecayedT> && _has_end_v<DecayedT> &&
                         _has_input_it_v<DecayedT>) {
      buffer += "{ ";
      auto it = value.begin();
      if (it != value.end()) {
        for (;;) {
          _append_selector(buffer, *it);
          if (++it == value.end())
            break;
          buffer += ", ";
        }
      }
      buffer += " }";
    } else if constexpr (_has_get_v<DecayedT> && _has_tuple_size_v<DecayedT>) {
      buffer += "< ";
      _append_tuple_impl(
          buffer, value,
          std::make_index_sequence<std::tuple_size_v<DecayedT>>{});
      buffer += " >";
    } else if constexpr (_has_ostream_insert_v<DecayedT>) {
      std::ostringstream oss;
      oss << value;
      buffer += oss.str();
    } else {
      static_assert(_always_false_v<T>,
                    "No valid stringification exists for the type.");
    }
  }

  template <class Tuple, std::size_t... Idx>
  static void _append_tuple_impl(std::string& buffer, const Tuple& value,
                                 std::index_sequence<Idx...>) {
    (((Idx == 0 ? "" : (buffer += ", ")),
      _append_selector(buffer, std::get<Idx>(value))),
     ...);
  }
};

struct Stringifier : public StringifierBase {};

}  // namespace internal

// ===================================================================
// --- ROBUST, DEFERRED-ACTION ARCHITECTURE ---
// ===================================================================
namespace internal {

/**
 * @class SinkConfig
 * @brief Stores the *intent* to create a sink.
 *
 * This is a lightweight, Plain Old Data (POD) style struct that is safe to
 * create globally or statically. It holds all the configuration for a single
 * log output (e.g., a file or `std::cout`). The actual sink resources (like
 * file handles) are only created later when the first log message is sent.
 */
class SinkConfig {
 public:
  SinkConfig& set_verbosity(Verbosity v) {
    verbosity = v;
    return *this;
  }

  SinkConfig& set_colors(Colors c) {
    colors = c;
    return *this;
  }

  SinkConfig& set_columns(const Columns& c) {
    columns = c;
    return *this;
  }

  Verbosity verbosity = Verbosity::INFO;
  Colors colors = Colors::DISABLE;
  Columns columns;
  OpenMode open_mode = OpenMode::REWRITE;
  std::variant<std::ostream*, std::filesystem::path> target;
};

/**
 * @class StreamHolder
 * @brief Manages the lifetime and thread-safety of a single output stream.
 *
 * This class abstracts away whether the output is a file (`std::ofstream`)
 * or a pre-existing stream (`std::ostream*`). It ensures that all writes
 * to the underlying stream are protected by a mutex.
 */
class StreamHolder {
  std::unique_ptr<std::ofstream> file_stream_;
  std::ostream* stream_ = nullptr;
  std::mutex mutex_;

 public:
  StreamHolder(const decltype(SinkConfig::target)& t, OpenMode m) {
    if (auto* os = std::get_if<std::ostream*>(&t)) {
      stream_ = *os;
    } else if (const auto* path = std::get_if<std::filesystem::path>(&t)) {
      // Ensure the directory exists before opening the file
      if (path->has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path->parent_path(), ec);
        // We can ignore the error, as ofstream will fail anyway.
      }
      auto mode = (m == OpenMode::APPEND) ? std::ios::app : std::ios::trunc;
      file_stream_ =
          std::make_unique<std::ofstream>(*path, std::ios::out | mode);
      stream_ = file_stream_.get();
    }
  }

  bool is_valid() const { return stream_ && stream_->good(); }

  void write(const std::string& msg) {
    const std::lock_guard lock(mutex_);
    if (is_valid())
      *stream_ << msg << std::flush;
  }
};

/**
 * @class Sink
 * @brief An active, fully-realized logging destination.
 *
 * It combines the configuration (`SinkConfig`) with a live stream resource
 * (`StreamHolder`). The `LoggerCore` manages a collection of these.
 */
class Sink {
 public:
  Sink(const SinkConfig& config)
      : config_(config),
        stream_holder_(
            std::make_shared<StreamHolder>(config.target, config.open_mode)) {}

  const SinkConfig& config() const { return config_; }

  void write(const std::string& formatted_message) {
    stream_holder_->write(formatted_message);
  }

 private:
  SinkConfig config_;
  std::shared_ptr<StreamHolder> stream_holder_;
};

/**
 * @class LoggerCore
 * @brief Singleton that manages the logger's global state.
 *
 * This class is the heart of the logger. It holds the list of sink
 * configurations and orchestrates their creation using `std::call_once` to
 * ensure thread-safety and avoid static initialization order issues. It also
 * tracks the program's start time for uptime calculation.
 */
class LoggerCore {
 public:
  static LoggerCore& instance() {
    static LoggerCore core;
    return core;
  }

  SinkConfig& add_config() {
    std::lock_guard lock(config_mutex_);
    configs_.emplace_back();
    return configs_.back();
  }

  const std::vector<std::shared_ptr<Sink>>& get_sinks() {
    std::call_once(sinks_initialized_flag_, [this] {
      std::lock_guard lock(config_mutex_);
      for (const auto& cfg : configs_) {
        sinks_.emplace_back(std::make_shared<Sink>(cfg));
      }
      // We can clear configs_ now as they've been used to build the sinks.
      configs_.clear();
    });
    return sinks_;
  }

  const std::chrono::steady_clock::time_point& get_start_time() const {
    return start_time_;
  }

 private:
  LoggerCore() = default;

  std::mutex config_mutex_;
  std::list<SinkConfig> configs_;
  std::once_flag sinks_initialized_flag_;
  std::vector<std::shared_ptr<Sink>> sinks_;
  const std::chrono::steady_clock::time_point start_time_{
      std::chrono::steady_clock::now()};
};

/// @brief Converts a Verbosity enum to its string representation.
inline const char* to_string(Verbosity v) {
  switch (v) {
    case Verbosity::ERR:
      return "ERR";
    case Verbosity::WARN:
      return "WARN";
    case Verbosity::NOTE:
      return "NOTE";
    case Verbosity::INFO:
      return "INFO";
    case Verbosity::DEBUG:
      return "DEBUG";
    case Verbosity::TRACE:
      return "TRACE";
  }
  return "???";
}

/// @brief Returns the ANSI color code for a given verbosity level.
inline const char* get_color_code(Verbosity v) {
  switch (v) {
    case Verbosity::ERR:
      return "\033[31m";  // Red
    case Verbosity::WARN:
      return "\033[33m";  // Yellow
    case Verbosity::NOTE:
      return "\033[36m";  // Cyan
    case Verbosity::INFO:
      return "\033[32m";  // Green
    case Verbosity::DEBUG:
      return "\033[35m";  // Magenta
    case Verbosity::TRACE:
      return "\033[90m";  // Bright Black (Gray)
    default:
      return "\033[0m";  // Reset
  }
}

constexpr const char* COLOR_RESET = "\033[0m";

/**
 * @brief The central function that receives, formats, and dispatches a log message.
 *
 * This function is the workhorse called by the `JIE_LOG_*` macros.
 * It performs several critical tasks:
 * 1. Prevents re-entrant logging from the same thread to avoid infinite loops.
 * 2. Lazily initializes sinks via `LoggerCore::get_sinks()` on the first call.
 * 3. Stringifies the user's variable arguments into a single message string.
 * 4. Iterates through all active sinks.
 * 5. For each sink, it checks the verbosity level.
 * 6. If the level is sufficient, it formats the complete log line according to
 *    the sink's unique `Columns` and `Colors` configuration.
 * 7. Dispatches the final, formatted string to the sink for writing.
 */
template <typename... Args>
void push_message(const Callsite& callsite, Verbosity verbosity,
                  Args&&... args) {
  // Prevent recursive logging (e.g., if a logged object's stringifier also logs).
  thread_local bool is_logging = false;
  if (is_logging)
    return;

  struct Guard {
    Guard() { is_logging = true; }

    ~Guard() { is_logging = false; }
  } guard;

  auto& sinks = LoggerCore::instance().get_sinks();
  if (sinks.empty())
    return;

  // Lazily stringify the user's message only if at least one sink needs it.
  std::string user_message;
  bool message_stringified = false;

  for (const auto& sink : sinks) {
    const auto& config = sink->config();

    if (verbosity <= config.verbosity) {
      if (!message_stringified) {
        user_message = Stringifier::stringify(std::forward<Args>(args)...);
        message_stringified = true;
      }

      std::stringstream ss;

      // --- Apply Formatting Columns ---
      if (config.colors == Colors::ENABLE) {
        ss << get_color_code(verbosity);
      }

      if (config.columns.datetime) {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;
        ss << '[' << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X")
           << '.' << std::setw(3) << std::setfill('0') << ms.count() << "] ";
      }

      if (config.columns.uptime) {
        auto uptime = std::chrono::duration<double>(
            std::chrono::steady_clock::now() -
            LoggerCore::instance().get_start_time());
        ss << '[' << std::fixed << std::setw(9) << std::setfill(' ')
           << std::setprecision(3) << uptime.count() << "s] ";
      }

      if (config.columns.thread) {
        ss << "[thread:" << std::this_thread::get_id() << "] ";
      }

      if (config.columns.level) {
        ss << '[' << to_string(verbosity) << "] ";
      }

      if (config.columns.callsite) {
        std::string_view file_sv = callsite.file;
        if (auto last_slash = file_sv.find_last_of("/\\");
            last_slash != std::string_view::npos) {
          file_sv.remove_prefix(last_slash + 1);
        }
        ss << '[' << file_sv << ':' << callsite.line << ' ' << callsite.function
           << "] ";
      }

      if (config.columns.message) {
        ss << user_message;
      }

      // Reset color after streaming user_message to color the messages.
      if (config.colors == Colors::ENABLE) {
        ss << COLOR_RESET;
      }

      ss << '\n';
      sink->write(ss.str());
    }
  }
}
}  // namespace internal

// ===================================================================
// --- PUBLIC API IMPLEMENTATION ---
// ===================================================================

/**
 * @brief Adds a file sink to the logger.
 * @param path The path to the log file. The directory will be created if it doesn't exist.
 * @param open_mode Whether to `REWRITE` or `APPEND` to the file.
 * @param verbosity The maximum verbosity level to log to this file.
 * @param columns A `Columns` struct to configure the output format.
 * @return A reference to the sink's configuration object, allowing for method chaining (e.g., `.set_verbosity(...)`).
 */
inline internal::SinkConfig& add_file_sink(
    const std::filesystem::path& path, OpenMode open_mode = OpenMode::REWRITE,
    Verbosity verbosity = Verbosity::TRACE, const Columns& columns = {}) {
  internal::SinkConfig& cfg = internal::LoggerCore::instance().add_config();
  cfg.target = path;
  cfg.open_mode = open_mode;
  cfg.set_verbosity(verbosity).set_colors(Colors::DISABLE).set_columns(columns);
  return cfg;
}

/**
 * @brief Adds an ostream sink (like `std::cout` or `std::cerr`) to the logger.
 * @param os The output stream to write to. The logger does NOT take ownership of the stream.
 * @param verbosity The maximum verbosity level to log to this stream.
 * @param colors Enable or disable ANSI colors for this stream.
 * @param columns A `Columns` struct to configure the output format.
 * @return A reference to the sink's configuration object, allowing for method chaining.
 */
inline internal::SinkConfig& add_ostream_sink(
    std::ostream& os, Verbosity verbosity = Verbosity::INFO,
    Colors colors = Colors::ENABLE, const Columns& columns = {}) {
  internal::SinkConfig& cfg = internal::LoggerCore::instance().add_config();
  cfg.target = &os;
  cfg.set_verbosity(verbosity).set_colors(colors).set_columns(columns);
  return cfg;
}

}  // namespace jie::log

// ===================================================================
// --- LOGGING MACROS (PUBLIC API) ---
// ===================================================================

#define JIE_LOG_ERR(...)                               \
  jie::log::internal::push_message(                    \
      {__FILE__, __LINE__, JIE_LOG_GET_FUNCTION_NAME}, \
      jie::log::Verbosity::ERR, __VA_ARGS__)

#define JIE_LOG_WARN(...)                              \
  jie::log::internal::push_message(                    \
      {__FILE__, __LINE__, JIE_LOG_GET_FUNCTION_NAME}, \
      jie::log::Verbosity::WARN, __VA_ARGS__)

#define JIE_LOG_NOTE(...)                              \
  jie::log::internal::push_message(                    \
      {__FILE__, __LINE__, JIE_LOG_GET_FUNCTION_NAME}, \
      jie::log::Verbosity::NOTE, __VA_ARGS__)

#define JIE_LOG_INFO(...)                              \
  jie::log::internal::push_message(                    \
      {__FILE__, __LINE__, JIE_LOG_GET_FUNCTION_NAME}, \
      jie::log::Verbosity::INFO, __VA_ARGS__)

#define JIE_LOG_DEBUG(...)                             \
  jie::log::internal::push_message(                    \
      {__FILE__, __LINE__, JIE_LOG_GET_FUNCTION_NAME}, \
      jie::log::Verbosity::DEBUG, __VA_ARGS__)

#define JIE_LOG_TRACE(...)                             \
  jie::log::internal::push_message(                    \
      {__FILE__, __LINE__, JIE_LOG_GET_FUNCTION_NAME}, \
      jie::log::Verbosity::TRACE, __VA_ARGS__)

// --- Debug-only logging macros ---
#ifdef _DEBUG
#define JIE_LOG_DERR(...) JIE_LOG_ERR(__VA_ARGS__)
#define JIE_LOG_DWARN(...) JIE_LOG_WARN(__VA_ARGS__)
#define JIE_LOG_DNOTE(...) JIE_LOG_NOTE(__VA_ARGS__)
#define JIE_LOG_DINFO(...) JIE_LOG_INFO(__VA_ARGS__)
#define JIE_LOG_DDEBUG(...) JIE_LOG_DEBUG(__VA_ARGS__)
#define JIE_LOG_DTRACE(...) JIE_LOG_TRACE(__VA_ARGS__)
#else
#define JIE_LOG_DERR(...) (void)0
#define JIE_LOG_DWARN(...) (void)0
#define JIE_LOG_DNOTE(...) (void)0
#define JIE_LOG_DINFO(...) (void)0
#define JIE_LOG_DDEBUG(...) (void)0
#define JIE_LOG_DTRACE(...) (void)0
#endif

#endif  // JIEHEADERGUARD_LOG_
#endif  // JIE_PICK_MODULES
