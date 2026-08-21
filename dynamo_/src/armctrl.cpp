// -*- C++ -*-
// armctrl.cpp

#include "armctrl.h"

#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <string>

#include "log.hpp"
#include "utils.h"

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kMinToSeconds = 60.0;
constexpr double kSecondToUseconds = 1e6;
}  // namespace

namespace eva::ctrl {

ArmCtrl::ArmCtrl(const std::string& device_path, uint32_t sleep_time_interval,
                 bool debug, bool enable_port)
    : ControlBase(device_path, enable_port),
      sleep_time_interval_(sleep_time_interval),
      debug_(debug) {
  if (!enable_port_) {
    JIE_LOG_INFO("ArmCtrl: hardware port disabled.");
    return;
  }

  // Check for environment variable override
  const char* env_path = std::getenv("ROBOT_DEVICE");
  if (env_path != nullptr && env_path[0] != '\0') {
    device_path_ = env_path;
  }
  // Auto-detect USB device if path is "auto" or contains "USB"
  else if (device_path == "auto" ||
           device_path.find("USB") != std::string::npos) {
    device_path_ = find_usb_device();
    if (device_path_.empty()) {
      JIE_LOG_ERR("ArmCtrl: No suitable USB device found. Disabling port.");
      enable_port_ = false;
    }
  }

  JIE_LOG_INFO("ArmCtrl: using device: ", device_path_);

  if (!is_port_open() && enable_port_) {
    open_port();
  }
}

void ArmCtrl::ctrl_motor(const int32_t motor_id, const double angle,
                         const int32_t velocity, const bool enable_multicontrol,
                         const bool verbose) {
  JIE_LOG_TRACE("ArmCtrl::ctrl_motor() - id:", motor_id, " angle:", angle,
                " vel:", velocity);

  uint8_t arg_motor_id = static_cast<uint8_t>(motor_id);
  uint8_t arg_dir = angle > 0.0 ? 0x01 : 0x00;

  // Velocity calculation
  int32_t arg_vel = (motor_id < kNumMotors)
                        ? static_cast<int32_t>(velocity * kReductionRatio)
                        : velocity;
  const uint8_t arg_lower_vel = static_cast<uint8_t>(arg_vel & 0xFF);
  const uint8_t arg_higher_vel = static_cast<uint8_t>((arg_vel >> 8) & 0xFF);

  // Acceleration
  const uint8_t arg_acc =
      (motor_id < kNumMotors)
          ? static_cast<uint8_t>(kDefaultAcceleration & 0xFF)
          : 0x00;

  // Angle calculation
  double effective_angle =
      (motor_id < kNumMotors) ? angle : angle / kReductionRatio;
  uint32_t arg_ang = static_cast<uint32_t>(std::fabs(effective_angle) *
                                           kMaxPulseValue / kTwoPi);
  const uint8_t arg_lowest_ang = static_cast<uint8_t>(arg_ang & 0xFF);
  const uint8_t arg_lower_ang = static_cast<uint8_t>((arg_ang >> 8) & 0xFF);
  const uint8_t arg_higher_ang = static_cast<uint8_t>((arg_ang >> 16) & 0xFF);
  const uint8_t arg_highest_ang = static_cast<uint8_t>((arg_ang >> 24) & 0xFF);
  const uint8_t arg_sign =
      static_cast<uint8_t>(static_cast<int32_t>(enable_multicontrol) & 0xFF);

  const uint8_t args[] = {arg_motor_id,
                          0xFD,
                          arg_dir,
                          arg_higher_vel,
                          arg_lower_vel,
                          arg_acc,
                          arg_highest_ang,
                          arg_higher_ang,
                          arg_lower_ang,
                          arg_lowest_ang,
                          0x00,
                          arg_sign,
                          0x6B};

  const std::string hex_string = hex_array_to_string(args, sizeof(args));
  JIE_LOG_NOTE("ArmCtrl::ctrl_motor() - Sent port message: ", hex_string);
  if (verbose) {
    std::cout << "ArmCtrl::ctrl_motor() - Sent: " << hex_string << '\n';
  }

  send(std::string(reinterpret_cast<const char*>(args), sizeof(args)));
}

void ArmCtrl::ctrl_motor_with_time(const int32_t motor_id, const double angle,
                                   const double duration,
                                   const bool enable_multicontrol,
                                   const bool verbose) {
  JIE_LOG_TRACE("ArmCtrl::ctrl_motor_with_time() - id:", motor_id,
                " angle:", angle, " duration:", duration);

  uint8_t arg_motor_id = static_cast<uint8_t>(motor_id);
  uint8_t arg_dir = angle > 0.0 ? 0x01 : 0x00;

  // Acceleration calculation
  const double acc = std::fabs(angle) * 4.0 / (duration * duration);
  const double acc_rpm = acc / kTwoPi * kMinToSeconds;

  double acc_param =
      (motor_id < kNumMotors)
          ? 1.0 / (acc_rpm / kSecondToUseconds * kReductionRatio * 50.0)
          : 1.0 / (acc_rpm / kSecondToUseconds * 50.0);
  const uint8_t acc_inter = static_cast<uint8_t>(
      std::lround(256.0 - std::clamp(acc_param, 1.0, 256.0)));
  const uint8_t arg_acc =
      (motor_id < kNumMotors) ? static_cast<uint8_t>(acc_inter & 0xFF) : 0x00;

  // Velocity calculation
  double velocity_rpm = acc_rpm * duration / 2.0;
  int32_t arg_vel =
      (motor_id < kNumMotors)
          ? static_cast<int32_t>(std::lround(velocity_rpm * kReductionRatio))
          : static_cast<int32_t>(velocity_rpm);
  const uint8_t arg_lower_vel = static_cast<uint8_t>(arg_vel & 0xFF);
  const uint8_t arg_higher_vel = static_cast<uint8_t>((arg_vel >> 8) & 0xFF);

  // Angle calculation
  const double effective_angle =
      (motor_id < kNumMotors) ? angle : angle / kReductionRatio;
  uint32_t arg_ang = static_cast<uint32_t>(std::fabs(effective_angle) *
                                           kMaxPulseValue / kTwoPi);
  const uint8_t arg_lowest_ang = static_cast<uint8_t>(arg_ang & 0xFF);
  const uint8_t arg_lower_ang = static_cast<uint8_t>((arg_ang >> 8) & 0xFF);
  const uint8_t arg_higher_ang = static_cast<uint8_t>((arg_ang >> 16) & 0xFF);
  const uint8_t arg_highest_ang = static_cast<uint8_t>((arg_ang >> 24) & 0xFF);
  const uint8_t arg_sign =
      static_cast<uint8_t>(static_cast<int32_t>(enable_multicontrol) & 0xFF);

  const uint8_t args[] = {arg_motor_id,
                          0xFD,
                          arg_dir,
                          arg_higher_vel,
                          arg_lower_vel,
                          arg_acc,
                          arg_highest_ang,
                          arg_higher_ang,
                          arg_lower_ang,
                          arg_lowest_ang,
                          0x00,
                          arg_sign,
                          0x6B};

  const std::string hex_string = hex_array_to_string(args, sizeof(args));
  JIE_LOG_NOTE("ArmCtrl::ctrl_motor_with_time() - Sent port message: ",
               hex_string);
  if (verbose) {
    std::cout << "ArmCtrl::ctrl_motor_with_time() - Sent: " << hex_string
              << '\n';
  }

  send(std::string(reinterpret_cast<const char*>(args), sizeof(args)));
}

void ArmCtrl::forward(const Kinematics::Thetas& input_thetas,
                      const int32_t velocity, const bool verbose) {
  if (debug_ || verbose)
    JIE_LOG_INFO("ArmCtrl::forward() triggered.");

  for (size_t i = 0; i < kNumMotors; ++i) {
    ctrl_motor(/*motor_id=*/i + 1, /*angle=*/input_thetas.values[i],
               /*velocity=*/velocity, /*enable_multicontrol=*/true, verbose);
    usleep(sleep_time_interval_);
  }

  uint8_t end_signal[] = {0x00, 0xFF, 0x66, 0x6B};
  send(std::string(reinterpret_cast<const char*>(end_signal),
                   sizeof(end_signal)));
  usleep(sleep_time_interval_ * 2);
}

void ArmCtrl::forward_with_velocities(const Kinematics::Thetas& input_thetas,
                                      const int32_t* velocities,
                                      const bool verbose) {
  if (debug_ || verbose)
    JIE_LOG_INFO("ArmCtrl::forward_with_velocities() triggered.");

  for (size_t i = 0; i < kNumMotors; ++i) {
    ctrl_motor(/*motor_id=*/i + 1, /*angle=*/input_thetas.values[i],
               /*velocity=*/velocities[i], /*enable_multicontrol=*/true,
               verbose);
    usleep(sleep_time_interval_);
  }

  uint8_t end_signal[] = {0x00, 0xFF, 0x66, 0x6B};
  send(std::string(reinterpret_cast<const char*>(end_signal),
                   sizeof(end_signal)));
  usleep(sleep_time_interval_ * 2);
}

void ArmCtrl::forward_with_time(const Kinematics::Thetas& input_thetas,
                                const double duration, const bool verbose) {
  if (debug_ || verbose)
    JIE_LOG_INFO("ArmCtrl::forward_with_time() triggered.");

  for (size_t i = 0; i < kNumMotors; ++i) {
    ctrl_motor_with_time(/*motor_id=*/i + 1, /*angle=*/input_thetas.values[i],
                         /*duration=*/duration, /*enable_multicontrol=*/true,
                         verbose);
    usleep(sleep_time_interval_);
  }

  uint8_t end_signal[] = {0x00, 0xFF, 0x66, 0x6B};
  send(std::string(reinterpret_cast<const char*>(end_signal),
                   sizeof(end_signal)));
  usleep(sleep_time_interval_ * 2);
}

void ArmCtrl::run_loop() {
  if (!is_port_open())
    throw std::runtime_error("Port not open");

  send("This is tty send test.\n");
  while (true) {
    try {
      std::string received = receive();
      if (!received.empty()) {
        printf("%s\n", received.c_str());
        send("OK!\n");
      }
    } catch (const std::exception& e) {
      fprintf(stderr, "Communication error: %s\n", e.what());
      break;
    }
  }
}

}  // namespace eva::ctrl
