// -*- C++ -*-
// instance.cpp
//
// High-level API orchestrator for Python bindings.

#include "instance_streamer.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "log.hpp"
#include "utils.h"

namespace {
constexpr double kUtimeToStime = 1e6;
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kRadToDeg = 57.295777754771045;
constexpr double kEpsilon = 1e-6;

int calculate_velocity(double delta_theta, unsigned int time_us) {
  const double kRadSecToRpm = 30.0 / kPi;
  double time_s = static_cast<double>(time_us) / kUtimeToStime;
  if (time_s < kEpsilon)
    return 1;
  double velocity_rad_s = std::fabs(delta_theta) / time_s;
  return std::max(1, static_cast<int>(velocity_rad_s * kRadSecToRpm));
}

}  // namespace

namespace eva::ctrl {

InstanceStreamer::InstanceStreamer(Kinematics* kinematics, ArmCtrl* armctrl,
                                   RotCtrl* rotctrl, PumpCtrl* pumpctrl,
                                   Kinematics::Thetas init_thetas,
                                   double init_theta, bool verbose, bool debug)
    : debug_(debug),
      verbose_(verbose),
      // BUG FIX: Initialize members directly from parameters before using them.
      init_thetas_(init_thetas),
      init_theta_(init_theta),
      thetas_(init_thetas),
      theta_(init_theta),
      kinematics_(kinematics),
      armctrl_(armctrl),
      rotctrl_(rotctrl),
      pumpctrl_(pumpctrl) {
  // Convert initial degrees from Python to radians for internal use.
  for (int i = 0; i < Kinematics::kNumJoints; ++i) {
    // BUG FIX: Use the 'init_thetas' parameter, not the uninitialized 'thetas_' member.
    init_thetas_.values[i] =
        std::fmod(init_thetas.values[i] / kRadToDeg, kTwoPi);
    thetas_.values[i] = init_thetas_.values[i];
  }
  init_theta_ = std::fmod(init_theta / kRadToDeg, kTwoPi);
  theta_ = init_theta_;

  // Set initial orientation
  kinematics_->solve_forward_kinematics(thetas_, orientation_, false);
  default_dev_path_ = armctrl_->get_device_path();
}

void InstanceStreamer::forward_kinematics(
    const Kinematics::Thetas& input_thetas,
    Kinematics::Orientation& output_orientation) {
  if (!kinematics_->solve_forward_kinematics(input_thetas, output_orientation,
                                             verbose_)) {
    JIE_LOG_ERR("InstanceStreamer::forward_kinematics() failed.");
  }
}

void InstanceStreamer::inverse_kinematics(
    const Kinematics::Orientation& input_orientation,
    Kinematics::Thetas& output_thetas) {
  Kinematics::Solutions solutions;
  if (kinematics_->solve_inverse_kinematics(input_orientation, thetas_,
                                            solutions, verbose_)) {
    for (int i = 0; i < Kinematics::kNumSolutions; ++i) {
      if (solutions.solution_flags[i]) {
        for (int j = 0; j < Kinematics::kNumJoints; ++j) {
          output_thetas.values[j] = solutions.solutions[i].values[j];
        }
        return;
      }
    }
  }
  JIE_LOG_ERR(
      "InstanceStreamer::inverse_kinematics() found no valid solution.");
}

void InstanceStreamer::dual_derive(Kinematics::Orientation* io_orientation,
                                   Kinematics::Thetas* io_thetas,
                                   const uint32_t duration_us,
                                   double** output_states,
                                   const bool use_orientation,
                                   const bool return_states) {
  if (io_orientation == nullptr || io_thetas == nullptr) {
    JIE_LOG_ERR("InstanceStreamer::dual_derive() requires non-null arguments.");
    return;
  }

  auto emit_with_time = [&](const Kinematics::Thetas& thetas) {
    Kinematics::Thetas inverted_thetas;
    for (int i = 0; i < Kinematics::kNumJoints; ++i) {
      inverted_thetas.values[i] =
          thetas.values[i] * static_cast<double>(invert_axes_[i]);
    }
    const double duration_s = static_cast<double>(duration_us) / kUtimeToStime;
    armctrl_->forward_with_time(inverted_thetas, duration_s, verbose_);
  };

  if (use_orientation) {
    JIE_LOG_NOTE("InstanceStreamer::dual_derive() - Solving IK...");
    Kinematics::Solutions solutions;
    kinematics_->solve_inverse_kinematics(*io_orientation, thetas_, solutions,
                                          verbose_);

    bool solution_found = false;
    for (int i = 0; i < Kinematics::kNumSolutions; ++i) {
      if (solutions.solution_flags[i]) {
        emit_with_time(solutions.solutions[i]);
        thetas_ = thetas_ + solutions.solutions[i];
        for (int j = 0; j < Kinematics::kNumJoints; ++j) {
          io_thetas->values[j] = solutions.solutions[i].values[j];
        }
        solution_found = true;
        break;
      }
    }
    if (!solution_found) {
      JIE_LOG_ERR("InstanceStreamer::dual_derive() - No IK solution found.");
      return;
    }
  } else {
    JIE_LOG_NOTE("InstanceStreamer::dual_derive() - Solving FK...");
    // Convert degrees to radians
    for (int i = 0; i < Kinematics::kNumJoints; ++i) {
      io_thetas->values[i] /= kRadToDeg;
    }
    emit_with_time(*io_thetas);
    thetas_ = thetas_ + *io_thetas;
  }

  // Update orientation and assign output values
  kinematics_->solve_forward_kinematics(thetas_, orientation_, false);
  *io_orientation = orientation_;

  if (return_states) {
    get_states(output_states, thetas_, verbose_);
  }
}

void InstanceStreamer::singular_derive(double angle) {
  rotctrl_->forward(angle / kRadToDeg, verbose_);
}

void InstanceStreamer::ctrl_rotation(const int32_t command_idx) {
  rotctrl_->forward_command(command_idx, false, verbose_);
}

void InstanceStreamer::ctrl_rotation_fallback(const int32_t command_idx) {
  rotctrl_->forward_command(command_idx, true, verbose_);
}

void InstanceStreamer::ctrl_pump(const int32_t command_idx) {
  pumpctrl_->forward(command_idx, verbose_);
}

std::vector<double> InstanceStreamer::get_direction_vector() {
  std::vector<double> dir_vec(3);
  orientation_to_direction_vector(orientation_.a, orientation_.b,
                                  orientation_.c, dir_vec[0], dir_vec[1],
                                  dir_vec[2]);
  return dir_vec;
}

void InstanceStreamer::restore() {
  thetas_ = init_thetas_;
  theta_ = init_theta_;
}

void InstanceStreamer::get_states(double** states,
                                  const Kinematics::Thetas& input_thetas,
                                  bool verbose) {
  if (!states) {
    JIE_LOG_WARN("InstanceStreamer::get_states() - 'states' is nullptr.");
    return;
  }

  // BUG FIX (REGRESSION): The original code converted degrees to radians here.
  // This logic was lost during refactoring and has been restored.
  Kinematics::Thetas rad_thetas;
  for (int i = 0; i < Kinematics::kNumJoints; ++i) {
    rad_thetas.values[i] =
        std::fmod(input_thetas.values[i] / kRadToDeg, kTwoPi);
  }

  kinematics_->get_states(states, rad_thetas, verbose);
}

double InstanceStreamer::get_rotary_theta() {
  // Convert internal radians back to degrees for the caller.
  return std::fmod(theta_ * kRadToDeg, 360.0);
}

Kinematics::Thetas InstanceStreamer::get_robot_arm_thetas() { return thetas_; }

Kinematics::Orientation InstanceStreamer::get_robot_arm_orientation() {
  return orientation_;
}

std::vector<int32_t> InstanceStreamer::get_invert_axes() {
  return invert_axes_;
}

}  // namespace eva::ctrl
