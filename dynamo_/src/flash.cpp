// -*- C++ -*-
// flash.cpp
//
// Modern C++ example for GPIO control using RAII.

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

// A simple RAII wrapper for a GPIO pin.
class GpioPin {
 public:
  explicit GpioPin(int pin) : pin_(pin) {
    export_pin();
    // Give the system a moment to create the GPIO directory
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  ~GpioPin() {
    try {
      unexport_pin();
    } catch (const std::exception& e) {
      std::cerr << "Error during GPIO unexport: " << e.what() << std::endl;
    }
  }

  // Delete copy operations to prevent double-unexporting.
  GpioPin(const GpioPin&) = delete;
  GpioPin& operator=(const GpioPin&) = delete;

  void set_direction(const std::string& direction) {
    write_to_sysfs("direction", direction);
  }

  void set_value(int value) { write_to_sysfs("value", std::to_string(value)); }

  int read_value() {
    std::string path = "/sys/class/gpio/gpio" + std::to_string(pin_) + "/value";
    std::ifstream value_file(path);
    if (!value_file.is_open()) {
      throw std::runtime_error("Failed to open pin value file for reading: " +
                               path);
    }
    char value;
    value_file >> value;
    return (value == '1') ? 1 : 0;
  }

 private:
  void write_to_sysfs(const std::string& attribute, const std::string& value) {
    std::string path =
        (attribute == "export" || attribute == "unexport")
            ? "/sys/class/gpio/" + attribute
            : "/sys/class/gpio/gpio" + std::to_string(pin_) + "/" + attribute;

    std::ofstream file(path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open GPIO file: " + path);
    }
    file << value;
  }

  void export_pin() {
    write_to_sysfs("export", std::to_string(pin_));
    std::cout << "Exported GPIO pin " << pin_ << std::endl;
  }

  void unexport_pin() {
    write_to_sysfs("unexport", std::to_string(pin_));
    std::cout << "Unexported GPIO pin " << pin_ << std::endl;
  }

  int pin_;
};

int main() {
  constexpr int kLedPin = 474;

  try {
    GpioPin led(kLedPin);
    led.set_direction("out");

    std::cout << "Setting pin " << kLedPin << " to HIGH." << std::endl;
    led.set_value(1);
    std::cout << "Pin " << kLedPin << " value: " << led.read_value()
              << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(3));

    std::cout << "Setting pin " << kLedPin << " to LOW." << std::endl;
    led.set_value(0);
    std::cout << "Pin " << kLedPin << " value: " << led.read_value()
              << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
