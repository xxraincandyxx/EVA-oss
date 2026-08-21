// -*- C++ -*-
// rotctrl.h

#ifndef EVA_ROTCTRL_H_
#define EVA_ROTCTRL_H_

#include <cstdint>
#include <string>

#include "base.h"

namespace eva::ctrl {

class RotCtrl : public ControlBase {
 public:
  enum class Command : uint8_t { kClamp, kRelease, kRotate };

  RotCtrl(const double& x, const double& y, const double& z,
          const double& init_theta, const std::string& device_path,
          bool debug = false, bool enable_port = true);

  ~RotCtrl() override { close_port(); }

  void forward(const double& angle, const bool verbose = false);
  void forward_command(const int32_t& command_idx, const bool use_fallback,
                       const bool verbose = false);

  double get_theta() const;
  void get_rotation_matrix(double* rotation_matrix) const;
  void get_target_position(double& x, double& y, double& z) const;
  void get_target_orientation(double& a, double& b, double& c) const;

 private:
  void ctrl_motor(const double& angle, const bool verbose = false);
  void ctrl_rotation(Command command, bool verbose = false);
  void ctrl_rotation_fallback(Command command, bool verbose = false);

  bool debug_;
  double x_center_, y_center_, z_center_, rot_theta_;
};

}  // namespace eva::ctrl

#endif  // EVA_ROTCTRL_H_
