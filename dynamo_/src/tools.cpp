// -*- C++ -*-
// tools.cpp

#include "utils.h"

#include <unistd.h>
#include <filesystem>
#include <iostream>

#include "log.hpp"

namespace fs = std::filesystem;
namespace logger = jie::log;

void init_logging(const std::string& subdir) {
  fs::path log_dir = fs::path(__FILE__).parent_path().parent_path() / "logs";
  if (!subdir.empty()) {
    log_dir /= subdir;
  }

  // Sink 1: Detailed debug log file.
  logger::add_file_sink(log_dir / "debug.log", logger::OpenMode::REWRITE)
      .set_verbosity(logger::Verbosity::TRACE);

  // Sink 2: Info-level log file for production monitoring.
  logger::add_file_sink(log_dir / "info.log", logger::OpenMode::APPEND)
      .set_verbosity(logger::Verbosity::INFO);

  // Sink 3: File with only the raw messages.
  logger::Columns message_only_cols;
  message_only_cols.datetime = false;
  message_only_cols.uptime = false;
  message_only_cols.thread = false;
  message_only_cols.level = false;
  message_only_cols.callsite = false;
  logger::add_file_sink(log_dir / "messages.log", logger::OpenMode::REWRITE,
                        logger::Verbosity::TRACE, message_only_cols);

  // Sink 4: Colorful console sink for live debugging.
  logger::add_ostream_sink(std::cout)
      .set_verbosity(logger::Verbosity::DEBUG)
      .set_colors(logger::Colors::ENABLE);
}

// REFACTORED: Renamed to snake_case for consistency.
std::string find_usb_device() {
  const char* candidates[] = {"/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyACM0",
                              "/dev/ttyAMA0"};
  for (const char* dev : candidates) {
    if (access(dev, F_OK) == 0) {
      JIE_LOG_INFO("Found USB device: ", dev);
      return dev;
    }
  }
  return "";
}
