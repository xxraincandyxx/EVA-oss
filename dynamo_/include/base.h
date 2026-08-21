// -*- C++ -*-
// base.h

#ifndef EVA_BASE_H_
#define EVA_BASE_H_

// Standard library headers
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

// Third-party headers

// Project headers
#include "log.hpp"

// Macros
#ifndef DEFAULT_BAUD
#define DEFAULT_BAUD B115200  // Defaults to 115200 baud
#endif

/* -------------------------------------------------------------------------- */

namespace eva::ctrl {

/**
 * @class ControlBase
 * @brief Base class for classes in communication with hardware devices.
 */
class ControlBase {
 public:
  // ---------------------------------------------------------------------------
  // CONSTRUCTORS AND DESTRUCTOR
  // ---------------------------------------------------------------------------

  // Rule: Constructors and destructors should be clearly defined.
  // The virtual destructor is essential for polymorphic base classes.
  // Add "= default" if the destructor is trivial.
  virtual ~ControlBase() = default;

  // Rule: Use the "= delete" keyword for the copy constructor and
  // assignment operator in base classes to prevent slicing/accidental copying.
  ControlBase(const ControlBase&) = delete;
  ControlBase& operator=(const ControlBase&) = delete;

  // ---------------------------------------------------------------------------
  // PUBLIC METHODS
  // ---------------------------------------------------------------------------

  // Rule: Pure virtual functions (interface methods) are common in base
  // classes. Functions should have clear, verbose names (e.g., GetValue rather
  // than GV). Use 'override' explicitly for methods that override a base class.
  // virtual void PureVirtualMethod(const std::string& input) = 0;

  // Rule: Inline small, non-complex functions (like simple getters) in the
  // class definition, but place complex functions in the .cc file.

  /**
   * @brief Getter for the module name.
   * @return The module name as a string.
   */
  std::string get_module_name() const { return module_name_; }

  /**
   * @brief Setter for the module name.
   * @param name The new module name.
   */
  void set_module_name(const std::string& name) { module_name_ = name; }

  /**
   * @brief Getter for the device path.
   * @return The device path as a string.
   */
  std::string get_device_path() const { return device_path_; }

  /**
   * @brief Getter for the port status.
   * @return True if the port is enabled, false otherwise.
   */
  bool get_port_status() const { return enable_port_; }

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PortControl& ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  virtual void open_port() {
    if (is_port_open()) {
      return;
    }

    // Try to open with read/write permissions
    fd_ = ::open(device_path_.c_str(), O_RDWR | O_NOCTTY);

    // Handle permission issues
    if (fd_ < 0 && errno == EACCES) {
      JIE_LOG_WARN("Permission denied. Attempting to adjust permissions...");
      std::string cmd = "sudo chmod a+rw " + device_path_;
      if (system(cmd.c_str()) != 0) {
        throw std::runtime_error("Permission adjustment failed");
      }
      fd_ = ::open(device_path_.c_str(), O_RDWR | O_NOCTTY);
    }

    if (fd_ < 0) {
      throw std::runtime_error("Failed to open " + device_path_ + ": " +
                               strerror(errno));
    }

    try {
      configure_port();
      JIE_LOG_INFO("Serial port configured successfully");
    } catch (...) {
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }

  virtual void close_port() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  virtual void send(const std::string& data) const {
    if (!this->enable_port_) {
      JIE_LOG_NOTE(
          "ArmCtrl::send() - Debug mode activated, port message sent but"
          " interrupted.");
      return;
    }

    if (!is_port_open()) {
      throw std::runtime_error("Port not open");
    }

    ssize_t bytesWritten = write(fd_, data.c_str(), /*nbyte=*/data.size());

    if (bytesWritten < 0) {
      throw std::runtime_error("Write failed");
    }
  }

  virtual std::string receive() const {
    if (!is_port_open()) {
      throw std::runtime_error("Port not open");
    }

    char buffer[BUFFER_SIZE];
    ssize_t bytesRead = read(fd_, buffer, /*nbyte=*/sizeof(buffer) - 1);

    if (bytesRead < 0) {
      throw std::runtime_error("Read failed");
    }

    if (bytesRead > 0) {
      buffer[bytesRead] = '\0';
      return std::string(buffer, bytesRead);
    }
    return "";
  }

  bool is_port_open() const noexcept { return fd_ >= 0; }

 protected:
  // Rule: Protected and private data members must end with an underscore.
  std::string module_name_{"ControlBase"};

  int32_t fd_ = -1;  // File descriptor for the serial port
  bool enable_port_{false};

  std::string device_path_;
  static constexpr size_t BUFFER_SIZE = 1024;

  // ---------------------------------------------------------------------------
  // PROTECTED METHODS
  // ---------------------------------------------------------------------------

  // Rule: Use protected methods for internal helper functions that derived
  // classes might need to access.
  ControlBase(const std::string& device_path, const bool& enable_port = false)
      : device_path_(device_path), enable_port_(enable_port) {}

  /* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PortControl& ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

  void configure_port() const {
    struct termios options;
    memset(&options, 0, sizeof(options));

    if (tcgetattr(fd_, &options) != 0) {
      throw std::runtime_error("tcgetattr failed: " +
                               std::string(strerror(errno)));
    }

    // Set baud rate
    cfsetispeed(&options, DEFAULT_BAUD);
    cfsetospeed(&options, DEFAULT_BAUD);

    // 8N1 configuration
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_cflag &= ~CRTSCTS;
    options.c_cflag |= CREAD | CLOCAL;

    // Raw input/output
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY | INLCR | ICRNL);
    options.c_oflag &= ~OPOST;

    // Timeouts: 100ms timeout, return immediately if any data
    options.c_cc[VTIME] = 1;
    options.c_cc[VMIN] = 0;

    if (tcsetattr(fd_, TCSANOW, &options) != 0) {
      throw std::runtime_error("tcsetattr failed: " +
                               std::string(strerror(errno)));
    }

    tcflush(fd_, TCIOFLUSH);
  }

  void HelperMethod();
};

}  // namespace eva::ctrl

#endif  // EVA_BASE_H_
