// -*- C++ -*-
// rotctrl.cpp

#include "rotctrl.h"

#include <unistd.h>
#include <cmath>
#include <functional>
#include <string>

#include "log.hpp"
#include "utils.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
// BUG FIX: Assume the rotary motor has the same pulse resolution as the arm motors.
// This constant was missing, leading to an incorrect pulse calculation.
constexpr double kMaxPulseValue = 160000.0;
}  // namespace

namespace eva::ctrl {

RotCtrl::RotCtrl(const double& x, const double& y, const double& z,
                 const double& init_theta, const std::string& device_path,
                 bool debug, bool enable_port)
    : ControlBase(device_path, enable_port),
      debug_(debug),
      x_center_(x),
      y_center_(y),
      z_center_(z),
      rot_theta_(init_theta) {
  if (!enable_port_) {
    JIE_LOG_INFO("RotCtrl: hardware port disabled.");
    return;
  }

  // Check for environment variable override
  const char* env_path = std::getenv("ROBOT_DEVICE");
  if (env_path != nullptr && env_path[0] != '\0') {
    device_path_ = env_path;
  }
  // Auto-detect USB device
  else if (device_path == "auto" ||
           device_path.find("USB") != std::string::npos) {
    device_path_ = find_usb_device();
    if (device_path_.empty()) {
      JIE_LOG_ERR("RotCtrl: No suitable device found. Disabling port.");
      enable_port_ = false;
    }
  }

  JIE_LOG_INFO("RotCtrl: using device: ", device_path_);

  if (!is_port_open() && enable_port_) {
    open_port();
  }
}

void RotCtrl::ctrl_motor(const double& angle, const bool verbose) {
  JIE_LOG_TRACE("RotCtrl::ctrl_motor() - angle:", angle);

  // BUG FIX: The original pulse calculation `fabs(angle) / PI` was physically incorrect.
  // Corrected to use the same logic as ArmCtrl: angle * (pulses_per_rev / rads_per_rev)
  uint32_t arg_ang =
      static_cast<uint32_t>(std::fabs(angle) * kMaxPulseValue / kTwoPi);

  uint8_t arg_lowest_ang = static_cast<uint8_t>(arg_ang & 0xFF);
  uint8_t arg_lower_ang = static_cast<uint8_t>((arg_ang >> 8) & 0xFF);
  uint8_t arg_higher_ang = static_cast<uint8_t>((arg_ang >> 16) & 0xFF);
  uint8_t arg_highest_ang = static_cast<uint8_t>((arg_ang >> 24) & 0xFF);

  uint8_t args[] = {0x28,
                    arg_lowest_ang,
                    arg_lower_ang,
                    arg_higher_ang,
                    arg_highest_ang,
                    0x29,
                    0x21};

  const std::string hex_string = hex_array_to_string(args, sizeof(args));
  JIE_LOG_NOTE("RotCtrl::ctrl_motor() - Sent port message: ", hex_string);
  if (verbose) {
    std::cout << "RotCtrl::ctrl_motor() - Sent: " << hex_string << '\n';
  }

  send(std::string(reinterpret_cast<const char*>(args), sizeof(args)));
}

void RotCtrl::ctrl_rotation(RotCtrl::Command command, const bool verbose) {
  JIE_LOG_TRACE("RotCtrl::ctrl_rotation() - command:",
                static_cast<int>(command));

  const uint8_t msg_clamp[] = {0x28, 0x01, 0x01, 0x80, 0x00, 0x01, 0x29, 0x21};
  const uint8_t msg_release[] = {0x28, 0x01, 0x01, 0x80,
                                 0x00, 0x00, 0x29, 0x21};
  const uint8_t msg_rotate[] = {0x28, 0x00, 0x00, 0x40, 0x00, 0x29, 0x21};

  auto send_message = [&](const uint8_t* data, size_t size) {
    const std::string hex_string = hex_array_to_string(data, size);
    JIE_LOG_NOTE("RotCtrl::ctrl_rotation() - Sent: ", hex_string);
    if (verbose)
      std::cout << "RotCtrl::ctrl_rotation() - Sent: " << hex_string << '\n';
    send(std::string(reinterpret_cast<const char*>(data), size));
  };

  switch (command) {
    case Command::kClamp:
      send_message(msg_clamp, sizeof(msg_clamp));
      break;
    case Command::kRelease:
      send_message(msg_release, sizeof(msg_release));
      break;
    case Command::kRotate:
      send_message(msg_rotate, sizeof(msg_rotate));
      break;
    default:
      JIE_LOG_WARN("RotCtrl::ctrl_rotation() - Unknown command.");
  }
}

void RotCtrl::ctrl_rotation_fallback(RotCtrl::Command command,
                                     const bool verbose) {
  JIE_LOG_TRACE("RotCtrl::ctrl_rotation_fallback() - command:",
                static_cast<int>(command));

  const uint8_t msg_clamp[] = {0x09, 0xFD, 0x01, 0x00, 0x20, 0x00, 0x00,
                               0x00, 0x0A, 0x00, 0x00, 0x00, 0x6B};
  const uint8_t msg_release[] = {0x09, 0xFD, 0x00, 0x00, 0x20, 0x00, 0x00,
                                 0x00, 0x0A, 0x00, 0x00, 0x00, 0x6B};
  const uint8_t msg_rotate_pre[] = {0x07, 0xFD, 0x01, 0x00, 0x0C, 0x00, 0x00,
                                    0x00, 0x01, 0x90, 0x00, 0x01, 0x6B};
  const uint8_t msg_rotate_in[] = {0x08, 0xFD, 0x00, 0x00, 0x0C, 0x00, 0x00,
                                   0x00, 0x01, 0x90, 0x00, 0x01, 0x6B};
  const uint8_t msg_rotate_suf[] = {0x00, 0xFF, 0x66, 0x6B};

  auto send_message = [&](const uint8_t* data, size_t size) {
    const std::string hex_string = hex_array_to_string(data, size);
    JIE_LOG_NOTE("RotCtrl::ctrl_rotation_fallback() - Sent: ", hex_string);
    if (verbose)
      std::cout << "RotCtrl::ctrl_rotation_fallback() - Sent: " << hex_string
                << '\n';
    send(std::string(reinterpret_cast<const char*>(data), size));
  };

  switch (command) {
    case Command::kClamp:
      send_message(msg_clamp, sizeof(msg_clamp));
      break;
    case Command::kRelease:
      send_message(msg_release, sizeof(msg_release));
      break;
    case Command::kRotate:
      send_message(msg_rotate_pre, sizeof(msg_rotate_pre));
      usleep(8000);
      send_message(msg_rotate_in, sizeof(msg_rotate_in));
      usleep(8000);
      send_message(msg_rotate_suf, sizeof(msg_rotate_suf));
      usleep(8000);
      break;
    default:
      JIE_LOG_WARN("RotCtrl::ctrl_rotation_fallback() - Unknown command.");
  }
}

void RotCtrl::forward(const double& angle, const bool verbose) {
  if (debug_ || verbose)
    JIE_LOG_INFO("RotCtrl::forward() - angle: ", angle);
  ctrl_motor(angle, verbose);
  rot_theta_ += angle;
  usleep(16000);
}

void RotCtrl::forward_command(const int32_t& command_idx,
                              const bool use_fallback, const bool verbose) {
  if (debug_ || verbose)
    JIE_LOG_INFO("RotCtrl::forward_command() - index: ", command_idx);

  Command command;
  switch (command_idx) {
    case 0:
      command = Command::kClamp;
      break;
    case 1:
      command = Command::kRelease;
      break;
    case 2:
      command = Command::kRotate;
      break;
    default:
      JIE_LOG_WARN("RotCtrl::forward_command() - Invalid command index: ",
                   command_idx);
      return;
  }

  if (!use_fallback) {
    ctrl_rotation(command, verbose);
  } else {
    ctrl_rotation_fallback(command, verbose);
  }
  usleep(16000);
}

double RotCtrl::get_theta() const { return rot_theta_; }

void RotCtrl::get_rotation_matrix(double* rotation_matrix) const {
  theta_to_rotation_matrix(rot_theta_, rotation_matrix);
}

void RotCtrl::get_target_position(double& x, double& y, double& z) const {
  x = x_center_;
  y = y_center_;
  z = z_center_;
}

void RotCtrl::get_target_orientation(double& a, double& b, double& c) const {
  double rotation_matrix[9];
  get_rotation_matrix(rotation_matrix);
  double euler_angles[3];
  rotation_matrix_to_euler_angles(rotation_matrix, euler_angles);
  a = euler_angles[0];
  b = euler_angles[1];
  c = euler_angles[2];
}

}  // namespace eva::ctrl
