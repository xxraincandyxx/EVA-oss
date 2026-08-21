// -*- C++ -*-
// instance_streamer.h

#ifndef EVA_INSTANCE_STREAMER_H_
#define EVA_INSTANCE_STREAMER_H_

#include <string>
#include <vector>

#include "armctrl.h"
#include "kinematics.h"
#include "pumpctrl.h"
#include "rotctrl.h"

namespace eva::ctrl {

class InstanceStreamer {
 public:
  InstanceStreamer(Kinematics* kinematics, ArmCtrl* armctrl, RotCtrl* rotctrl,
                   PumpCtrl* pumpctrl, Kinematics::Thetas init_thetas,
                   double init_theta = 0.0, bool verbose = false,
                   bool debug = false);
  virtual ~InstanceStreamer() = default;

  void forward_kinematics(const Kinematics::Thetas& input_thetas,
                          Kinematics::Orientation& output_orientation);

  void inverse_kinematics(const Kinematics::Orientation& input_orientation,
                          Kinematics::Thetas& output_thetas);

  // BUG FIX: Renamed u_duration to duration_us for consistency with the .cpp file.
  void dual_derive(Kinematics::Orientation* io_orientation,
                   Kinematics::Thetas* io_thetas,
                   uint32_t duration_us = 2400000,
                   double** output_states = nullptr,
                   bool use_orientation = false, bool return_states = false);

  void singular_derive(double angle);

  void ctrl_rotation(int32_t command_idx);
  void ctrl_rotation_fallback(int32_t command_idx);
  void ctrl_pump(int32_t command_idx);

  std::vector<double> get_direction_vector();
  void restore();
  void get_states(double** states, const Kinematics::Thetas& input_thetas,
                  bool verbose = false);

  double get_rotary_theta();
  Kinematics::Thetas get_robot_arm_thetas();
  Kinematics::Orientation get_robot_arm_orientation();
  std::vector<int32_t> get_invert_axes();

 private:
  bool debug_;
  bool verbose_;
  std::string default_dev_path_;

  Kinematics::Thetas init_thetas_;
  double init_theta_;

  Kinematics::Thetas thetas_;
  Kinematics::Orientation orientation_;
  double theta_;

  const std::vector<int32_t> invert_axes_{1, 1, -1, -1, -1, -1};

  // Pointers to underlying hardware controllers (ownership managed externally).
  Kinematics* kinematics_;
  ArmCtrl* armctrl_;
  RotCtrl* rotctrl_;
  PumpCtrl* pumpctrl_;
};

}  // namespace eva::ctrl

#endif  // EVA_INSTANCE_STREAMER_H_
