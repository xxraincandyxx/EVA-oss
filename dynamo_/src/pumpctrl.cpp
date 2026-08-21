// -*- C++ -*-
// pumpctrl.cpp

#include <unistd.h>
#include <string>

#include "pumpctrl.h"

#include "log.hpp"
#include "utils.h"

namespace eva::ctrl {

PumpCtrl::PumpCtrl(const std::string& device_path, bool debug, bool enable_port)
    : ControlBase(device_path, enable_port),
      debug_(debug),
      status_(Status::kDetached) {
  if (!enable_port_) {
    JIE_LOG_INFO("PumpCtrl: hardware port disabled.");
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
      JIE_LOG_ERR("PumpCtrl: No suitable device found. Disabling port.");
      enable_port_ = false;
    }
  }

  JIE_LOG_INFO("PumpCtrl: using device: ", device_path_);

  if (!is_port_open() && enable_port_) {
    open_port();
  }
}

void PumpCtrl::ctrl_pump(Command command, const bool verbose) {
  JIE_LOG_TRACE("PumpCtrl::ctrl_pump() - command:", static_cast<int>(command));

  const uint8_t msg_attach[] = {0x28, 0x02, 0x29, 0x21};
  const uint8_t msg_detach[] = {0x28, 0x04, 0x29, 0x21};
  const uint8_t msg_shutdown[] = {0x28, 0x03, 0x29, 0x21};

  auto send_message = [&](const uint8_t* data, size_t size) {
    const std::string hex_string = hex_array_to_string(data, size);
    JIE_LOG_NOTE("PumpCtrl::ctrl_pump() - Sent: ", hex_string);
    if (verbose) {
      std::cout << "PumpCtrl::ctrl_pump() - Sent: " << hex_string << '\n';
    }
    send(std::string(reinterpret_cast<const char*>(data), size));
  };

  switch (command) {
    case Command::kAttach:
      send_message(msg_attach, sizeof(msg_attach));
      status_ = Status::kAttached;
      break;
    case Command::kDetach:
      send_message(msg_detach, sizeof(msg_detach));
      status_ = Status::kDetached;
      break;
    case Command::kShutdown:
      send_message(msg_shutdown, sizeof(msg_shutdown));
      status_ = Status::kDetached;
      break;
    default:
      JIE_LOG_WARN("PumpCtrl::ctrl_pump() - Unknown command.");
  }
}

void PumpCtrl::forward(const int32_t& command_idx, bool verbose) {
  if (debug_ || verbose) {
    JIE_LOG_INFO("PumpCtrl::forward() - index: ", command_idx);
  }

  Command command;
  switch (command_idx) {
    case 0:
      command = Command::kAttach;
      break;
    case 1:
      command = Command::kDetach;
      break;
    case 2:
      command = Command::kShutdown;
      break;
    default:
      JIE_LOG_WARN("PumpCtrl::forward() - Invalid command index: ",
                   command_idx);
      return;
  }

  ctrl_pump(command, verbose);
  usleep(16000);
}

PumpCtrl::Status PumpCtrl::get_status() const { return status_; }

}  // namespace eva::ctrl
