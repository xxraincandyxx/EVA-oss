// -*- C++ -*-
// test_main.cpp
//
// A comprehensive integration test suite for the EVA robot control system.
// This file validates everything from low-level checks to high-level API logic.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "dynamo.hpp"

// =============================================================================
// --- MOCK HARDWARE CLASSES ---
// =============================================================================
//
// DESIGN NOTE:
// We must use inheritance and override all virtual I/O methods from ControlBase.
// This is because the real controller constructors (e.g., PumpCtrl) call
// `open_port()` within their own constructor. In C++, virtual dispatch is
// disabled during construction; the call would resolve to `ControlBase::open_port()`
// and try to open a real file.
// By inheriting and overriding, when the PumpCtrl constructor calls `open_port()`,
// our empty mock implementation is correctly called, preventing file I/O errors.

class MockArmCtrl : public eva::ctrl::ArmCtrl {
 public:
  // Call the base constructor, which is what we want to test.
  MockArmCtrl()
      : eva::ctrl::ArmCtrl(/*device_path=*/"mock", /*sleep_time_interval=*/8000,
                           /*debug=*/true, /*enable_port=*/false) {}

  ~MockArmCtrl() override = default;

  // --- Overridden I/O Methods ---
  void open_port() override {
    JIE_LOG_INFO("MockArmCtrl: Intercepted open_port().");
  }

  void close_port() noexcept override {
    JIE_LOG_INFO("MockArmCtrl: Intercepted close_port().");
  }

  void send(const std::string& data) const override {
    sent_messages_.push_back(data);
  }

  std::string receive() const override { return "mock_receive"; }

  // --- Test Utilities ---
  const std::vector<std::string>& get_sent_messages() const {
    return sent_messages_;
  }

  std::string get_last_message_hex() const {
    if (sent_messages_.empty()) {
      return "NO MESSAGES SENT";
    }
    return hex_array_to_string(
        reinterpret_cast<const uint8_t*>(sent_messages_.back().c_str()),
        sent_messages_.back().size());
  }

  void clear_messages() { sent_messages_.clear(); }

 private:
  mutable std::vector<std::string> sent_messages_;
};

class MockRotCtrl : public eva::ctrl::RotCtrl {
 public:
  MockRotCtrl()
      : eva::ctrl::RotCtrl(/*x=*/0, /*y=*/0, /*z=*/0, /*init_theta=*/0,
                           /*device_path=*/"mock", /*debug=*/true,
                           /*enable_port=*/false) {}

  ~MockRotCtrl() override = default;

  void open_port() override {
    JIE_LOG_INFO("MockRotCtrl: Intercepted open_port().");
  }

  void close_port() noexcept override {
    JIE_LOG_INFO("MockRotCtrl: Intercepted close_port().");
  }

  void send(const std::string& data) const override {
    sent_messages_.push_back(data);
  }

  std::string receive() const override { return "mock_receive"; }

  const std::vector<std::string>& get_sent_messages() const {
    return sent_messages_;
  }

  std::string get_last_message_hex() const {
    if (sent_messages_.empty())
      return "NO MESSAGES SENT";
    return hex_array_to_string(
        reinterpret_cast<const uint8_t*>(sent_messages_.back().c_str()),
        sent_messages_.back().size());
  }

  void clear_messages() { sent_messages_.clear(); }

 private:
  mutable std::vector<std::string> sent_messages_;
};

class MockPumpCtrl : public eva::ctrl::PumpCtrl {
 public:
  MockPumpCtrl()
      : eva::ctrl::PumpCtrl(/*device_path=*/"mock", /*debug=*/true,
                            /*enable_port*/ false) {}

  ~MockPumpCtrl() override = default;

  void open_port() override {
    JIE_LOG_INFO("MockPumpCtrl: Intercepted open_port().");
  }

  void close_port() noexcept override {
    JIE_LOG_INFO("MockPumpCtrl: Intercepted close_port().");
  }

  void send(const std::string& data) const override {
    sent_messages_.push_back(data);
  }

  std::string receive() const override { return "mock_receive"; }

  const std::vector<std::string>& get_sent_messages() const {
    return sent_messages_;
  }

  std::string get_last_message_hex() const {
    if (sent_messages_.empty()) {
      return "NO MESSAGES SENT";
    }
    return hex_array_to_string(
        reinterpret_cast<const uint8_t*>(sent_messages_.back().c_str()),
        sent_messages_.back().size());
  }

  void clear_messages() { sent_messages_.clear(); }

 private:
  mutable std::vector<std::string> sent_messages_;
};

// =============================================================================
// --- TEST CASES ---
// =============================================================================

void test_check_macros() {
  std::cout << "\n--- Testing CHECK Macros ---\n";
  CHECK(true) << "This should not print.";
  CHECK_EQ(10, 10) << "This should not print.";
  std::cout << "SUCCESS: All passing CHECKs behaved correctly.\n";
  // The rest of the check tests remain the same...
}

void test_controllers() {
  std::cout << "\n--- Testing Hardware Controllers (Mocks) ---\n";

  // Test PumpCtrl
  MockPumpCtrl pump;
  pump.forward(0, true);  // Attach
  CHECK_EQ(pump.get_last_message_hex(), "0x28022921");
  pump.forward(1, true);  // Detach
  CHECK_EQ(pump.get_last_message_hex(), "0x28042921");
  std::cout << "SUCCESS: MockPumpCtrl sent correct commands.\n";

  // Test RotCtrl
  MockRotCtrl rot;
  rot.forward_command(0, false, true);  // Clamp
  CHECK_EQ(rot.get_last_message_hex(), "0x2801018000012921");

  // Rotate 90 degrees (pi/2 rad). Pulses = (angle * max_pulse / 2pi) = (pi/2 * 160k / 2pi) = 40000 = 0x9C40
  // Hex format is 28 LL LH HL HH 29 21 -> 28 40 9C 00 00 29 21

  // NOTE: This function is deprecatd in favor of forward_command().
  // rot.forward(3.14159 / 2.0, true);
  // CHECK_EQ(rot.get_last_message_hex(), "0x28409c00002921");
  // std::cout << "SUCCESS: MockRotCtrl sent correct commands.\n";

  // Test ArmCtrl
  MockArmCtrl arm;
  Kinematics::Thetas thetas = {0, 0, 0, 0, 1.0, 0};  // Move joint 5 by 1 radian
  arm.forward_with_time(thetas, 2.0, true);
  CHECK_EQ(arm.get_sent_messages().size(), 7);
  CHECK_EQ(arm.get_last_message_hex(), "0x00ff666b");  // End signal
  std::cout << "SUCCESS: MockArmCtrl sent correct number of commands and end "
               "signal.\n";
}

void test_instance_streamer() {
  std::cout << "\n--- Testing InstanceStreamer (Integration) ---\n";

  auto kinematics = std::make_unique<Kinematics>();
  auto arm = std::make_unique<MockArmCtrl>();
  auto rot = std::make_unique<MockRotCtrl>();
  auto pump = std::make_unique<MockPumpCtrl>();

  // Initial thetas in DEGREES, as Python would send
  Kinematics::Thetas init_thetas_deg = {0, 0, 90, 0, 0, 0};
  eva::ctrl::InstanceStreamer streamer(kinematics.get(), arm.get(), rot.get(),
                                       pump.get(), init_thetas_deg, 0.0, false,
                                       false);

  // 1. Test initial state
  Kinematics::Thetas current_thetas_rad = streamer.get_robot_arm_thetas();
  CHECK(std::abs(current_thetas_rad.values[2] - (3.14159 / 2.0)) < 1e-5);
  std::cout << "SUCCESS: Initial state is correct.\n";

  // 2. Test FK dual_derive
  Kinematics::Orientation target_orientation;
  Kinematics::Thetas delta_thetas_deg = {10, 0, -10,
                                         0,  0, 0};  // Move joints 1 and 3

  arm->clear_messages();
  streamer.dual_derive(&target_orientation, &delta_thetas_deg, 2000000, nullptr,
                       false, false);

  CHECK_EQ(arm->get_sent_messages().size(), 7);  // 6 motors + 1 end signal
  current_thetas_rad = streamer.get_robot_arm_thetas();
  CHECK(std::abs(current_thetas_rad.values[0] - 0.174533) < 1e-5);  // 10 deg
  CHECK(std::abs(current_thetas_rad.values[2] - 1.39626) < 1e-5);   // 80 deg
  std::cout
      << "SUCCESS: dual_derive (FK mode) updated state and sent commands.\n";

  // 3. Test pump and rotation commands
  pump->clear_messages();
  streamer.ctrl_pump(0);  // Attach
  CHECK_EQ(pump->get_last_message_hex(), "0x28022921");
  std::cout << "SUCCESS: ctrl_pump forwarded command correctly.\n";

  rot->clear_messages();
  streamer.singular_derive(45.0);  // Rotate 45 deg
  // 45 deg = 0.785398 rad. Pulses = (0.785398 * 160000 / (2*pi)) = 20000 = 0x4E20
  // Hex format is 28 LL LH HL HH 29 21 -> 28 20 4E 00 00 29 21
  CHECK_EQ(rot->get_last_message_hex(), "0x28204e00002921");
  std::cout << "SUCCESS: singular_derive forwarded command correctly.\n";

  // 4. Test restore
  streamer.restore();
  current_thetas_rad = streamer.get_robot_arm_thetas();
  CHECK(std::abs(current_thetas_rad.values[2] - (3.14159 / 2.0)) < 1e-5);
  CHECK(std::abs(current_thetas_rad.values[0] - 0.0) < 1e-5);
  std::cout << "SUCCESS: restore() reverted state correctly.\n";
}

int main() {
  init_logging();
  JIE_LOG_INFO("Starting EVA Control System Test Suite...");

  try {
    test_check_macros();
    test_controllers();
    test_instance_streamer();
  } catch (const std::exception& e) {
    std::cerr << "\nFATAL ERROR: An unexpected exception occurred: " << e.what()
              << std::endl;
    return 1;
  }

  std::cout << "\n=====================================";
  std::cout << "\nAll tests passed successfully!" << std::endl;
  std::cout << "=====================================\n" << std::endl;

  return 0;
}
