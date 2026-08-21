// -*- C++ -*-
// armctrl.h

#ifndef EVA_ARMCTRL_H_
#define EVA_ARMCTRL_H_

#include <cstdint>
#include <string>

#include "base.h"
#include "kinematics.h"

namespace eva::ctrl {

class ArmCtrl : public ControlBase {
 public:
  ArmCtrl(const std::string& device_path = "/dev/ttyHS1",
          uint32_t sleep_time_interval = 8000, bool debug = false,
          bool enable_port = true);

  ~ArmCtrl() override { close_port(); }

  void forward(const Kinematics::Thetas& input_thetas, int32_t velocity = 5,
               bool verbose = false);
  void forward_with_velocities(const Kinematics::Thetas& input_thetas,
                               const int32_t* velocities, bool verbose = false);
  void forward_with_time(const Kinematics::Thetas& input_thetas,
                         double duration, bool verbose = false);

  void run_loop();  // For self initial test

  constexpr static int32_t kNumMotors = 6;
  constexpr static uint8_t kDefaultAcceleration = 224;
  constexpr static double kMaxPulseValue = 160000.0;
  constexpr static double kReductionRatio = 50.0;

 private:
  void ctrl_motor(int32_t motor_id, double angle, int32_t velocity,
                  bool enable_multicontrol = false, bool verbose = false);
  void ctrl_motor_with_time(int32_t motor_id, double angle, double duration,
                            bool enable_multicontrol, bool verbose);

  const uint32_t sleep_time_interval_;
  bool debug_;
};

}  // namespace eva::ctrl

#endif  // EVA_ARMCTRL_H_
