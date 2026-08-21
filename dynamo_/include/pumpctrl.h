// -*- C++ -*-
// pumpctrl.h

#ifndef EVA_PUMPCTRL_H_
#define EVA_PUMPCTRL_H_

#include <cstdint>
#include <string>

#include "base.h"

namespace eva::ctrl {

class PumpCtrl : public ControlBase {
 public:
  enum class Status : uint8_t { kAttached, kDetached };
  enum class Command : uint8_t { kAttach, kDetach, kShutdown };

  PumpCtrl(const std::string& device_path = "/dev/ttyHS1", bool debug = false,
           bool enable_port = true);

  ~PumpCtrl() override { close_port(); }

  void forward(const int32_t& command_idx, bool verbose = false);

  Status get_status() const;

 private:
  void ctrl_pump(Command command, bool verbose = false);

  bool debug_;
  Status status_;
};

}  // namespace eva::ctrl

#endif  // EVA_PUMPCTRL_H_
