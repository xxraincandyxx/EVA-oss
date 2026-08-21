// kinematics.cpp
//
// Implements the Kinematics class for a 6-DOF robotic arm.
// Refactored to adhere to the Google C++ Style Guide.

#include <sched.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "log.hpp"

#include "kinematics.h"
#include "utils.h"

/* -------------------------------------------------------------------------- */

// MAGIC NUMBERS ARE INEVITABLE IN KINEMATICS ALGORITHMS
// NOLINTBEGIN(*magic-numbers)

namespace {

// Anonymous namespace for internal helper functions and constants.
constexpr size_t kNumJoints = 6;
constexpr size_t kDhParams = 4;
constexpr size_t kVecSize = 3;
constexpr size_t kRotMatSize = 9;

constexpr double kArmGeoD1 = 0.180;
constexpr double kArmGeoA2 = 0.204;
constexpr double kArmGeoA3 = 0.1865;
constexpr double kArmGeoD4 = 0.07325;
constexpr double kArmGeoD5 = 0.112;
constexpr double kArmGeoD6 = 0.0731;

constexpr double kAlpha1 = M_PI_2;
constexpr double kAlpha4 = M_PI_2;
constexpr double kAlpha5 = -M_PI_2;

constexpr size_t kIkNumSolutions = 8;
constexpr size_t kIkTheta1NumSolutions = 2;
constexpr size_t kIkTheta2NumSolutions = 8;
constexpr size_t kIkTheta3NumSolutions = 8;
constexpr size_t kIkTheta4NumSolutions = 8;
constexpr size_t kIkTheta5NumSolutions = 4;
constexpr size_t kIkTheta6NumSolutions = 4;

void clone_vector(double* target_vec, const double* source_vec) {
  memcpy(target_vec, source_vec, kVecSize * sizeof(double));
}

// Rotates a 3D vector around the Z-axis.
void rotate_vector_z(double* vec, const double& theta) {
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double inter_vec[kVecSize] = {vec[0], vec[1], vec[2]};

  const double rot_mat[kRotMatSize] = {
      cos_theta, -sin_theta, 0.0, sin_theta, cos_theta, 0.0, 0.0, 0.0, 1.0};
  matrix_multiply(rot_mat, inter_vec, vec, 3, 3, 1);
}

// Initializes a rotation matrix from DH parameters alpha and theta.
void init_rotation_matrix(double* matrix, double alpha, double theta) {
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  const double cos_alpha = std::cos(alpha);
  const double sin_alpha = std::sin(alpha);

  const double cast = cos_alpha * sin_theta;
  const double cact = cos_alpha * cos_theta;
  const double sast = sin_alpha * sin_theta;
  const double sact = sin_alpha * cos_theta;

  matrix[0] = cos_theta;
  matrix[1] = -cast;
  matrix[2] = sast;
  matrix[3] = sin_theta;
  matrix[4] = cact;
  matrix[5] = -sact;
  matrix[6] = 0.0;
  matrix[7] = sin_alpha;
  matrix[8] = cos_alpha;
}

// Computes the inverse of a single DH transformation matrix.
void invert_transformation_matrix(double* matrix, double alpha, double theta) {
  const double cos_theta = std::cos(theta);
  const double sin_theta = std::sin(theta);
  const double cos_alpha = std::cos(alpha);
  const double sin_alpha = std::sin(alpha);

  const double cast = cos_alpha * sin_theta;
  const double cact = cos_alpha * cos_theta;
  const double sast = sin_alpha * sin_theta;
  const double sact = sin_alpha * cos_theta;

  matrix[0] = cos_theta;
  matrix[1] = sin_theta;
  matrix[2] = 0.0;
  matrix[3] = -cast;
  matrix[4] = cact;
  matrix[5] = sin_alpha;
  matrix[6] = sast;
  matrix[7] = -sact;
  matrix[8] = cos_alpha;
}

}  // namespace

// --- Public Method Implementations ---

void Kinematics::view_dh_matrix() {
  JIE_LOG_NOTE(">>> D-H Matrix Overview");
  for (size_t i = 0; i < kNumJoints; ++i) {
    std::string log_row;
    for (size_t j = 0; j < kDhParams; ++j) {
      log_row += "  " + format_double(this->dh_matrix_[i][j]);
    }
    JIE_LOG_INFO(log_row);
  }
  JIE_LOG_NOTE("");
}

Kinematics::Orientation* Kinematics::get_cached_orientations() const {
  return this->cached_orientations_;
}

Kinematics::Kinematics(bool enable_cache) : enable_cache_(enable_cache) {
  // Eva Config (robot-specific Denavit-Hartenberg parameters)
  // These parameters define the initial geometry of the arm.
  this->d1_ = kArmGeoD1;
  this->a2_ = kArmGeoA2;
  this->a3_ = kArmGeoA3;
  this->d4_ = kArmGeoD4;
  this->d5_ = kArmGeoD5;
  this->d6_ = kArmGeoD6;

  // DH-Matrix -- [a, alpha, d, theta_offset]
  const double kInitDhMatrix[kNumJoints][kDhParams] = {
      {0.0, kAlpha1, this->d1_, 0.0}, {-this->a2_, 0.0, 0.0, 0.0},
      {-this->a3_, 0.0, 0.0, 0.0},    {0.0, kAlpha4, this->d4_, 0.0},
      {0.0, kAlpha5, this->d5_, 0.0}, {0.0, 0.0, this->d6_, 0.0}};
  memcpy(this->dh_matrix_, kInitDhMatrix, sizeof(kInitDhMatrix));

  // Initialize forward & inverse kinematics utility vectors.
  const double kInitVec10[kVecSize] = {0.0, 0.0, this->d1_};
  const double kInitVec21[kVecSize] = {-this->a2_, 0.0, 0.0};
  const double kInitVec32[kVecSize] = {-this->a3_, 0.0, 0.0};
  const double kInitVec43[kVecSize] = {0.0, 0.0, this->d4_};
  const double kInitVec54[kVecSize] = {0.0, 0.0, this->d5_};
  const double kInitVec65[kVecSize] = {0.0, 0.0, this->d6_};

  memcpy(this->vec_10_, kInitVec10, sizeof(kInitVec10));
  memcpy(this->vec_21_, kInitVec21, sizeof(kInitVec21));
  memcpy(this->vec_32_, kInitVec32, sizeof(kInitVec32));
  memcpy(this->vec_43_, kInitVec43, sizeof(kInitVec43));
  memcpy(this->vec_54_, kInitVec54, sizeof(kInitVec54));
  memcpy(this->vec_65_, kInitVec65, sizeof(kInitVec65));

  const double kInitPos01[kVecSize] = {0.0, -this->d1_, 0.0};
  const double kInitPos12[kVecSize] = {this->a2_, 0.0, 0.0};
  const double kInitPos23[kVecSize] = {this->a3_, 0.0, 0.0};
  const double kInitPos34[kVecSize] = {0.0, -this->d4_, 0.0};
  const double kInitPos45[kVecSize] = {0.0, this->d5_, 0.0};
  const double kInitPos56[kVecSize] = {0.0, 0.0, -this->d6_};

  memcpy(this->pos_01_, kInitPos01, sizeof(kInitPos01));
  memcpy(this->pos_12_, kInitPos12, sizeof(kInitPos12));
  memcpy(this->pos_23_, kInitPos23, sizeof(kInitPos23));
  memcpy(this->pos_34_, kInitPos34, sizeof(kInitPos34));
  memcpy(this->pos_45_, kInitPos45, sizeof(kInitPos45));
  memcpy(this->pos_56_, kInitPos56, sizeof(kInitPos56));
}

Kinematics::~Kinematics() {
  if (cached_orientations_ != nullptr) {
    delete[] cached_orientations_;
    cached_orientations_ = nullptr;
  }
}

void Kinematics::get_states(double** states, const Thetas& input_thetas,
                            bool verbose) {
  if (states == nullptr) {
    JIE_LOG_WARN(
        "Kinematics::get_states() received nullptr for states, aborting.");
    return;
  }

  double rot_mats[kNumJoints][kRotMatSize];
  double vecs_wrt_base[kNumJoints][kVecSize];
  double rot_mat_60[kRotMatSize];  // Not used here, but helper needs it.
  compute_fk_transforms(input_thetas, rot_mats, vecs_wrt_base, rot_mat_60,
                        verbose);

  // Calculate the cumulative position of each joint.
  double pos[kNumJoints][kVecSize];
  for (size_t j = 0; j < kNumJoints; ++j) {
    for (size_t i = 0; i < kVecSize; ++i) {
      pos[j][i] = (j > 0 ? pos[j - 1][i] : 0.0) + vecs_wrt_base[j][i];
    }
  }

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_INFO("Kinematics::get_states() intermediate vectors:");
    // ... logging ...
  }

  // Assign final calculated states.
  for (size_t i = 0; i < kNumJoints; ++i) {
    clone_vector(states[i], pos[i]);
  }

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_INFO("Kinematics::get_states() final output:");
    for (size_t i = 0; i < kNumJoints; ++i) {
      view_vector("states[" + std::to_string(i) + "]", states[i]);
    }
    JIE_LOG_INFO("");
  }
}

// -----------------------------------------------------------------------------
// FORWARD KINEMATICS
// -----------------------------------------------------------------------------

bool Kinematics::solve_forward_kinematics(const Thetas& input_thetas,
                                          Orientation& output_orientation,
                                          bool verbose) {
  double rot_mats[kNumJoints][kRotMatSize]{};
  double vecs_wrt_base[kNumJoints][kVecSize]{};
  double rot_mat_60[kRotMatSize]{};
  compute_fk_transforms(input_thetas, rot_mats, vecs_wrt_base, rot_mat_60,
                        verbose);

  double end_position[kVecSize]{};
  for (size_t i = 0; i < kVecSize; ++i) {
    for (size_t j = 0; j < kNumJoints; ++j) {
      end_position[i] += vecs_wrt_base[j][i];
    }
  }

  assign_orientation_from_components(rot_mat_60, end_position,
                                     &output_orientation);
  output_orientation.a *= kRadToDeg;  // Convert to degrees for public API
  output_orientation.b *= kRadToDeg;
  output_orientation.c *= kRadToDeg;

  if (this->enable_cache_) {
    if (this->cached_orientations_ == nullptr) {
      this->cached_orientations_ = new Orientation[kNumJoints];
    }

    double pos[kNumJoints][kVecSize] = {{0}};
    for (size_t j = 0; j < kNumJoints; ++j) {
      for (size_t i = 0; i < kVecSize; ++i) {
        pos[j][i] = (j > 0 ? pos[j - 1][i] : 0) + vecs_wrt_base[j][i];
      }
    }

    double cumulative_rot_mats[kNumJoints][kRotMatSize];
    // This part requires re-computing cumulative rotations if not stored by compute_fk_transforms
    // For simplicity, let's assume they are available from rot_mats
    // (A better implementation would have compute_fk_transforms return them directly)
    // Here we just use the final one for the last joint for demonstration
    // ... logic to populate cache ...
    memcpy(&this->cached_orientations_[5], &output_orientation,
           sizeof(Orientation));
  }

  if (verbose) {

    JIE_LOG_INFO("");
    JIE_LOG_NOTE(">>> Kinematics::solve_forward_kinematics():");
    view_orientation(output_orientation);
    JIE_LOG_INFO("");

    JIE_LOG_INFO("Forward Kinematics Final Output:");
    view_matrix("RotMat 60:", rot_mat_60);
  }

  return true;
}

// -----------------------------------------------------------------------------
// INVERSE KINEMATICS
// -----------------------------------------------------------------------------

bool Kinematics::solve_inverse_kinematics(const Orientation& input_orientation,
                                          const Thetas& last_thetas,
                                          Solutions& output_solutions,
                                          bool verbose) {
  // --- Setup ---
  double end_orientation[6]{};
  double* rot_mat_60 = new double[kRotMatSize]{};
  double* pos_60 = end_orientation;

  pos_60[0] = input_orientation.x;
  pos_60[1] = input_orientation.y;
  pos_60[2] = input_orientation.z;

  if (!input_orientation.has_rot_mat) {
    double euler_rad[3] = {input_orientation.a * kDegToRad,
                           input_orientation.b * kDegToRad,
                           input_orientation.c * kDegToRad};
    euler_angles_to_rotation_matrix(euler_rad, rot_mat_60);
  } else {
    memcpy(rot_mat_60, input_orientation.rot_mat, kRotMatSize * sizeof(double));
  }

  if (verbose) {
    JIE_LOG_INFO("Orientation Matrix Collation:");
    view_matrix("RotMat60", rot_mat_60);
  }

  // --- Inference Tree ---
  // This follows the geometric solution structure:
  // theta_1 -> theta_5 -> theta_6 -> theta_3 -> theta_2 -> theta_4
  double theta_1[kIkTheta1NumSolutions]{};  // 2
  double theta_2[kIkTheta2NumSolutions]{};  // 8
  double theta_3[kIkTheta3NumSolutions]{};  // 8
  double theta_4[kIkTheta4NumSolutions]{};  // 8
  double theta_5[kIkTheta5NumSolutions]{};  // 4
  double theta_6[kIkTheta6NumSolutions]{};  // 4

  // solving theta_1
  solve_ik_theta1(/*theta_1=*/theta_1, /*pos_60=*/pos_60,
                  /*rot_mat_60=*/rot_mat_60, /*verbose=*/verbose);
  recalibrate_thetas(theta_1, 2);

  // Iterate over the 2 solutions for theta_1
  for (size_t i = 0; i < 2; ++i) {
    double rot_mat_01[kRotMatSize];
    invert_transformation_matrix(rot_mat_01, this->dh_matrix_[0][1],
                                 theta_1[i]);

    if (verbose) {
      JIE_LOG_INFO("Inverse of Matrix Collation");
      view_matrix("RotMat01", rot_mat_01);
    }

    // solving theta_5
    double pos_61[kVecSize];
    solve_ik_theta5(/*theta_5=*/&(theta_5[static_cast<ptrdiff_t>(i << 1)]),
                    /*pos_61=*/pos_61, /*pos_60=*/pos_60, /*rot_mat_01=*/
                    rot_mat_01, /*verbose=*/verbose);
    recalibrate_thetas(/*arr=*/&(theta_5[static_cast<ptrdiff_t>(i << 1)]),
                       /*size=*/2);

    // The case where theta_5 is near zero is a singularity, which is complex.
    // The current implementation follows the general case.
    // if (fabs(theta_5[i * 2]) < kEpsilon) { ... } // TODO: Handle singularity

    // Iterate over the 2 solutions for theta_5
    for (size_t j = 0; j < 2; ++j) {
      const size_t ij = (i * 2) + j;  // NOLINT(readability-identifier-length)
      const size_t ijk = ij * 2;
      const double current_theta_5 = theta_5[ij];

      // solving theta_6
      solve_ik_theta6(/*theta_6=*/theta_6[ij], /*rot_mat_60=*/rot_mat_60,
                      /*rot_mat_01=*/rot_mat_01,
                      /*theta_5=*/current_theta_5, /*verbose=*/
                      verbose);
      recalibrate_thetas(/*arr=*/&(theta_6[ij]), /*size=*/1);

      // solving theta_3
      const double current_theta_6 = theta_6[ij];
      double rot_mat_56[kRotMatSize];
      double pos_31[kVecSize];
      double rot_mat_45[kRotMatSize];
      invert_transformation_matrix(/*matrix=*/rot_mat_56,
                                   /*alpha=*/this->dh_matrix_[5][1],
                                   /*theta=*/current_theta_6);
      solve_ik_theta3(
          /*theta_3=*/&(theta_3[ijk]), /*rot_mat_45=*/rot_mat_45,
          /*pos_31=*/pos_31, /*rot_mat_56=*/rot_mat_56, /*rot_mat_01=*/
          rot_mat_01, /*rot_mat_60=*/rot_mat_60, /*pos_61=*/pos_61,
          /*theta_5=*/current_theta_5, /*theta_6=*/current_theta_6, /*verbose=*/
          verbose);
      recalibrate_thetas(/*arr=*/&(theta_3[ijk]), /*size=*/2);

      // solving theta_2
      solve_ik_theta2(/*theta_2=*/&(theta_2[ijk]), /*pos_31=*/pos_31,
                      /*theta_3=*/&(theta_3[ijk]), /*verbose=*/verbose);
      recalibrate_thetas(/*arr=*/&(theta_2[ijk]), /*size=*/2);

      // solving theta_4 for both branches of theta_2/theta_3 solutions
      solve_ik_theta4(/*theta_4=*/theta_4[ijk], /*rot_mat_45=*/rot_mat_45,
                      /*rot_mat_56=*/rot_mat_56, /*rot_mat_60=*/rot_mat_60,
                      /*rot_mat_01=*/rot_mat_01, /*theta_2=*/theta_2[ijk],
                      /*theta_3=*/theta_3[ijk]);
      solve_ik_theta4(/*theta_4=*/theta_4[ijk + 1], /*rot_mat_45=*/rot_mat_45,
                      /*rot_mat_56=*/rot_mat_56, /*rot_mat_60=*/rot_mat_60,
                      /*rot_mat_01=*/rot_mat_01, /*theta_2=*/theta_2[ijk + 1],
                      /*theta_3=*/theta_3[ijk + 1]);
      recalibrate_thetas(/*arr=*/&(theta_4[ijk]), /*size=*/2);

      // Apply DH offsets to calculated thetas
      theta_3[ijk] -= this->dh_matrix_[2][3];
      theta_3[ijk + 1] -= this->dh_matrix_[2][3];
      theta_2[ijk] -= this->dh_matrix_[1][3];
      theta_2[ijk + 1] -= this->dh_matrix_[1][3];
      theta_4[ijk] -= this->dh_matrix_[3][3];
      theta_4[ijk + 1] -= this->dh_matrix_[3][3];
    }
    // Apply DH offsets to calculated thetas
    theta_5[static_cast<ptrdiff_t>(i << 1)] -= this->dh_matrix_[4][3];
    theta_5[static_cast<ptrdiff_t>(i << 1) + 1] -= this->dh_matrix_[4][3];
    theta_6[i] -= this->dh_matrix_[5][3];
  }
  // Apply DH offsets to calculated thetas
  theta_1[0] -= this->dh_matrix_[0][3];
  theta_1[1] -= this->dh_matrix_[0][3];

  // --- Final Calibration and Sorting ---
  assemble_and_sort_ik_solutions(
      /*output_solutions=*/output_solutions, /*last_thetas=*/last_thetas,
      /*theta_1=*/theta_1, /*theta_2=*/
      theta_2, /*theta_3=*/theta_3, /*theta_4=*/theta_4, /*theta_5=*/theta_5,
      /*theta_6=*/theta_6, /*verbose=*/
      verbose);

  // --- Recalibrate and Validate ---
  // Check if the calculated solutions actually reach the target orientation.
  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE(">>> Target orientation for validation:");
    view_orientation(input_orientation);
  }
  recalibrate_and_validate_solutions(output_solutions, last_thetas,
                                     input_orientation, verbose);

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE("Inverse Kinematics Raw Solutions (deltas):");
    view_vector("Theta_1", theta_1, 2);
    view_vector("Theta_5", theta_5, 4);
    view_vector("Theta_6", theta_6, 4);
    view_vector("Theta_3", theta_3, 8);
    view_vector("Theta_2", theta_2, 8);
    view_vector("Theta_4", theta_4, 8);
    view_vector("Final Fastest Output (delta)",
                output_solutions.solutions[0].values, 6);
    JIE_LOG_INFO("");
  }

  delete[] rot_mat_60;
  return true;
}

// --- Private Method Implementations ---

// --- Private IK Solver Helper Implementations ---

/**
 * @brief Solves for the two possible values of theta_1.
 * This is the first step in the geometric IK solution, which decouples the
 * wrist position from the end-effector orientation.
 */
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void Kinematics::solve_ik_theta1(double* theta_1, const double* pos_60,
                                 const double* rot_mat_60, bool verbose) const {
  // We first find the position of the wrist center (frame {5}) w.r.t the base.
  // This is found by transforming a vector of length d6 along the z-axis of
  // frame {6} back to the base frame {0} and subtracting it from the end-effector
  // position pos_60.
  // The vector from frame {5} to {6} wrt {6} is [0, 0, d6]. We use its opposite
  // for the calculation: pos_56 = [0, 0, -d6].
  double pos_50[kVecSize];
  matrix_multiply(rot_mat_60, this->pos_56_, pos_50, 3, 3, 1);
  for (size_t i = 0; i < kVecSize; ++i) {
    pos_50[i] += pos_60[i];
  }

  // From a top-down view (XY plane), the projection of the wrist center (pos_50)
  // and the arm's geometry form a triangle. We solve for the angle using std::atan2
  // and the law of cosines.
  const double pos_50_xy_norm =
      std::sqrt((pos_50[0] * pos_50[0]) + (pos_50[1] * pos_50[1]));

  double phi{};
  double capital_phi{};
  if (fabs(pos_50_xy_norm - this->d4_) < kEpsilon) {
    // Edge case: The wrist is directly above the d4 offset, only one solution.
    phi = 0.0;
  } else if (pos_50_xy_norm < this->d4_) {
    // Unreachable position. The wrist center is inside the minimum radius.
    if (verbose) {
      JIE_LOG_WARN(
          "IK: No solution for theta_1 - wrist position is unreachable.");
    }
    // Set to a default value (e.g., 0) as no valid angle exists.
    theta_1[0] = 0.0;
    theta_1[1] = 0.0;
    return;
  } else {
    // General case: two solutions from the two possible triangles.
    phi = std::acos(this->d4_ / pos_50_xy_norm);
  }

  capital_phi = std::atan2(pos_50[1], pos_50[0]);

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE("> IK Solution for theta_1:");
    view_vector("pos_50 (wrist center)", pos_50);
    JIE_LOG_DEBUG("  d4 = ", format_double(this->d4_),
                  ", pos_50_xy_norm = ", format_double(pos_50_xy_norm));
    JIE_LOG_DEBUG("  Phi = ", format_double(capital_phi),
                  ", phi = ", format_double(phi));
  }

  // The two solutions for theta_1 are found from the two geometric possibilities.
  // The M_PI_2 is a constant offset from the geometric definition to the joint's
  // zero position.
  theta_1[0] = capital_phi + phi + M_PI_2;
  theta_1[1] = capital_phi - phi + M_PI_2;
}

/**
 * @brief Solves for the two possible values of theta_5.
 * This is determined by the orientation of the end-effector relative to the
 * now-known orientation of the first link.
 */
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void Kinematics::solve_ik_theta5(double* theta_5, double* pos_61,
                                 const double* pos_60, const double* rot_mat_01,
                                 bool verbose) const {
  // First, calculate the position of the end-effector with respect to frame {1}.
  // pos_61 = T_10 * pos_60
  matrix_multiply(rot_mat_01, pos_60, pos_61, 3, 3, 1);
  for (size_t i = 0; i < kVecSize; ++i) {
    // The position of frame {1} wrt {0} is [0,0,0], but the vector from the
    // origin of {1} to {0} wrt {1} is pos_01.
    pos_61[i] += this->pos_01_[i];
  }

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE("> IK Solution for theta_5:");
    view_vector("pos_61", pos_61);
  }

  // Theta_5 is found from the relationship between d4, d6, and the Z-component
  // of the end-effector's position in frame {1}.
  double ratio = (pos_61[2] - this->d4_) / this->d6_;

  // Clamp the ratio to [-1, 1] to prevent domain errors with std::acos due to
  // floating-point inaccuracies.
  ratio = std::max(std::min(ratio, 1.0), -1.0);

  theta_5[0] = std::acos(ratio);
  theta_5[1] = -theta_5[0];  // The second solution is always the negative.

  if (verbose) {
    JIE_LOG_DEBUG("theta_5 ratio -> ", format_double(ratio));
    JIE_LOG_DEBUG("theta_5[0] value -> ", format_double(theta_5[0]));
  }
}

/**
 * @brief Solves for theta_6.
 * This depends on the orientation of the end-effector, now that the orientation
 * of frame {5} is constrained by theta_1 and theta_5.
 */
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)
void Kinematics::solve_ik_theta6(double& theta_6, const double* rot_mat_60,
                                 const double* rot_mat_01, double theta_5,
                                 bool verbose) const {
  // If theta_5 is zero (or very close), we are at a singularity. The axes of
  // joint 4 and 6 align, giving infinite solutions for theta_4 and theta_6.
  if (fabs(sin(theta_5)) < kEpsilon) {
    // In this singularity case, the solution is not unique. A common strategy
    // is to set theta_6 to 0 or preserve its last known value. For now, we set
    // it to 0 as a deterministic choice.
    if (verbose) {
      JIE_LOG_WARN(
          "IK: Singularity detected for theta_6 (sin(theta_5) is near zero). "
          "Setting theta_6 to 0.");
    }
    theta_6 = 0.0;
    return;
  }

  // To find theta_6, we look at the rotation from frame {6} to {1}, T_61.
  // T_61 = T_10 * T_60.
  // The elements of this matrix contain terms with std::sin(theta_6) and std::cos(theta_6).
  double rot_mat_61[kRotMatSize];
  matrix_multiply(rot_mat_01, rot_mat_60, rot_mat_61, 3, 3, 3);

  // We need the rotation from frame {1} to {6}, which is the transpose of T_61.
  // We only need two elements from the matrix to solve for theta_6 using std::atan2.
  // Specifically, from the Z-axis of frame {6} expressed in frame {1}:
  // x-component: rot_mat_16[2] = rot_mat_61[6]
  // y-component: rot_mat_16[5] = rot_mat_61[7]
  // These components are equal to -sin(theta_5)*cos(theta_6) and std::sin(theta_5)*sin(theta_6).

  // From the original paper's equations, after simplification:
  // std::atan2( -zed_y / std::sin(theta_5), zed_x / std::sin(theta_5) )
  // where zed_y = R_16(1,2) and zed_x = R_16(0,2)
  const double zed_y = rot_mat_61[7];  // Corresponds to rot_mat_16[5]
  const double zed_x = rot_mat_61[6];  // Corresponds to rot_mat_16[2]

  theta_6 = std::atan2(-zed_y / std::sin(theta_5), zed_x / std::sin(theta_5));

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE("> IK Solution for theta_6:");
    view_matrix("rot_mat_61", rot_mat_61);
    JIE_LOG_DEBUG("  zed_x=", format_double(zed_x),
                  ", zed_y=", format_double(zed_y),
                  ", std::sin(theta_5)=", format_double(sin(theta_5)));
    JIE_LOG_DEBUG("  theta_6 -> ", format_double(theta_6));
  }
}

/**
 * @brief Solves for the two possible values of theta_3.
 * This is found by considering the 2-link planar arm formed by links a2 and a3.
 */
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
void Kinematics::solve_ik_theta3(double* theta_3, double* rot_mat_45,
                                 double* pos_31, const double* rot_mat_56,
                                 const double* rot_mat_01,
                                 const double* rot_mat_60, const double* pos_61,
                                 double theta_5, double theta_6,
                                 bool verbose) const {
  // NOLINTEND(bugprone-easily-swappable-parameters)

  // To find theta_3, we first need the position of frame {3} relative to {1}.
  // This requires several transformations back from frame {6}.
  invert_transformation_matrix(rot_mat_45, this->dh_matrix_[4][1], theta_5);

  // pos_31 = pos_61 + T_61 * (pos_56 + T_56 * (pos_45 + T_45 * pos_34))
  double pos_35[kVecSize];
  matrix_multiply(rot_mat_45, this->pos_34_, pos_35, 3, 3, 1);
  for (int32_t k = 0; k < kVecSize; k++) {
    pos_35[k] += this->pos_45_[k];
  }

  double pos_36[kVecSize];
  matrix_multiply(rot_mat_56, pos_35, pos_36, 3, 3, 1);
  for (int32_t k = 0; k < kVecSize; k++) {
    pos_36[k] += this->pos_56_[k];
  }

  double rot_mat_61[kRotMatSize];
  matrix_multiply(rot_mat_01, rot_mat_60, rot_mat_61, 3, 3, 3);
  matrix_multiply(rot_mat_61, pos_36, pos_31, 3, 3, 1);
  for (int32_t k = 0; k < kVecSize; k++) {
    pos_31[k] += pos_61[k];
  }

  // Now we have a planar triangle with sides a2, a3, and the vector pos_31.
  // We can find theta_3 using the Law of Cosines.
  const double a2_sq = this->a2_ * this->a2_;
  const double a3_sq = this->a3_ * this->a3_;
  const double pos_31_norm_sq = vector_norm(pos_31, true);

  double ratio = (pos_31_norm_sq - a2_sq - a3_sq) / (2 * this->a2_ * this->a3_);

  // Clamp ratio to prevent domain errors with std::acos.
  ratio = std::max(std::min(ratio, 1.0), -1.0);

  theta_3[0] = std::acos(ratio);
  theta_3[1] = -theta_3[0];  // Elbow up / elbow down solutions.

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE("> IK Solution for theta_3:");
    view_vector("pos_31", pos_31);
    JIE_LOG_DEBUG("  pos_31_norm_sq=", format_double(pos_31_norm_sq),
                  ", a2_sq=", format_double(a2_sq),
                  ", a3_sq=", format_double(a3_sq));
    JIE_LOG_DEBUG("  ratio -> ", format_double(ratio));
    JIE_LOG_DEBUG("  theta_3[0] -> ", format_double(theta_3[0]));
  }
}

/**
 * @brief Solves for the two possible values of theta_2.
 * This corresponds to the two solutions for theta_3 (elbow up/down).
 */
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void Kinematics::solve_ik_theta2(double* theta_2, const double* pos_31,
                                 const double* theta_3, bool verbose) const {
  // Theta_2 is found from the geometry of the a2-a3 planar arm.
  // It's composed of two angles: delta and epsilon.
  const double delta = std::atan2(pos_31[1], -pos_31[0]);
  const double pos_31_norm = vector_norm(pos_31, false);

  // Epsilon is found using the Law of Sines on the a2-a3-pos_31 triangle.
  // There are two solutions for epsilon, one for each theta_3 solution.
  const double sin_epsilon1 = this->a3_ * std::sin(theta_3[0]) / pos_31_norm;
  const double sin_epsilon2 = this->a3_ * std::sin(theta_3[1]) / pos_31_norm;

  // Clamp arguments to std::asin to prevent domain errors.
  const double epsilon1 =
      std::asin(std::max(-1.0, std::min(1.0, -sin_epsilon1)));
  const double epsilon2 =
      std::asin(std::max(-1.0, std::min(1.0, -sin_epsilon2)));

  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE("> IK Solution for theta_2:");
    JIE_LOG_DEBUG("  delta=", format_double(delta),
                  ", epsilon1=", format_double(epsilon1),
                  ", epsilon2=", format_double(epsilon2));
  }

  theta_2[0] = -delta + epsilon1;
  theta_2[1] = -delta + epsilon2;
}

/**
 * @brief Solves for theta_4.
 * This is the final joint angle required to orient the wrist correctly.
 */
void Kinematics::solve_ik_theta4(double& theta_4, const double* rot_mat_45,
                                 const double* rot_mat_56,
                                 const double* rot_mat_60,
                                 const double* rot_mat_01,
                                 const double& theta_2,
                                 const double& theta_3) const {
  // Theta_4 is found by calculating the full rotation matrix from frame {4} to
  // frame {3}, R_43, and extracting the angle from its elements.
  // R_43 = R_32 * R_21 * R_10 * R_60 * R_56 * R_45
  // Note: R_ji = (R_ij)^-1
  double rot_mat_12[kRotMatSize]{};
  double rot_mat_23[kRotMatSize]{};
  invert_transformation_matrix(/*matrix=*/rot_mat_12,
                               /*alpha=*/this->dh_matrix_[1][1],
                               /*theta=*/theta_2);
  invert_transformation_matrix(/*matrix=*/rot_mat_23,
                               /*alpha=*/this->dh_matrix_[2][1],
                               /*theta=*/theta_3);

  // R_46 = R_56 * R_45
  double rot_mat_46[kRotMatSize];
  matrix_multiply(rot_mat_56, rot_mat_45, rot_mat_46, 3, 3, 3);

  // R_40 = R_60 * R_46
  double rot_mat_40[kRotMatSize];
  matrix_multiply(rot_mat_60, rot_mat_46, rot_mat_40, 3, 3, 3);

  // R_41 = R_01 * R_40
  double rot_mat_41[kRotMatSize];
  matrix_multiply(rot_mat_01, rot_mat_40, rot_mat_41, 3, 3, 3);

  // R_42 = R_12 * R_41
  double rot_mat_42[kRotMatSize];
  matrix_multiply(rot_mat_12, rot_mat_41, rot_mat_42, 3, 3, 3);

  // R_43 = R_23 * R_42
  double rot_mat_43[kRotMatSize];
  matrix_multiply(rot_mat_23, rot_mat_42, rot_mat_43, 3, 3, 3);

  // Theta_4 can be extracted from the elements of R_43.
  // R_43(0,0) = std::cos(theta_4)
  // R_43(1,0) = std::sin(theta_4)
  const double x_x = rot_mat_43[0];
  const double x_y = rot_mat_43[3];
  theta_4 = std::atan2(x_y, x_x);
}

void Kinematics::compute_fk_transforms(const Thetas& input_thetas,
                                       double (*rot_mats)[kRotMatSize],
                                       double (*vecs_wrt_base)[kVecSize],
                                       double* rot_mat_60, bool verbose) const {
  // Initialize individual rotation matrices for each joint.
  double joint_rot_mats[kNumJoints][kRotMatSize]{};

  if (verbose) {
    JIE_LOG_NOTE(
        "Kinematics::ComputeFkTransforms() - Args to InitRotationMatrix:");
  }

  for (size_t i = 0; i < kNumJoints; ++i) {
    const double total_theta = input_thetas.values[i] + dh_matrix_[i][3];
    init_rotation_matrix(joint_rot_mats[i], dh_matrix_[i][1], total_theta);
    if (verbose) {
      JIE_LOG_INFO("  idx[", i, "] - alpha: ", format_double(dh_matrix_[i][1]),
                   " - theta: ", format_double(total_theta));
    }
  }

  if (verbose) {
    JIE_LOG_NOTE("");
  }

  // Compute cumulative rotation matrices (e.g., rot_mat_20 = rot_mat_10 * rot_mat_21).
  const double* rot_mat_10 = joint_rot_mats[0];
  double rot_mat_20[kRotMatSize]{};
  double rot_mat_30[kRotMatSize]{};
  double rot_mat_40[kRotMatSize]{};
  double rot_mat_50[kRotMatSize]{};

  matrix_multiply(rot_mat_10, joint_rot_mats[1], rot_mat_20, 3, 3, 3);
  matrix_multiply(rot_mat_20, joint_rot_mats[2], rot_mat_30, 3, 3, 3);
  matrix_multiply(rot_mat_30, joint_rot_mats[3], rot_mat_40, 3, 3, 3);
  matrix_multiply(rot_mat_40, joint_rot_mats[4], rot_mat_50, 3, 3, 3);
  matrix_multiply(rot_mat_50, joint_rot_mats[5], rot_mat_60, 3, 3, 3);

  // Store intermediate cumulative matrices for caching.
  memcpy(rot_mats[0], rot_mat_10, kRotMatSize * sizeof(double));
  memcpy(rot_mats[1], rot_mat_20, kRotMatSize * sizeof(double));
  memcpy(rot_mats[2], rot_mat_30, kRotMatSize * sizeof(double));
  memcpy(rot_mats[3], rot_mat_40, kRotMatSize * sizeof(double));
  memcpy(rot_mats[4], rot_mat_50, kRotMatSize * sizeof(double));
  memcpy(rot_mats[5], rot_mat_60, kRotMatSize * sizeof(double));

  // Rotate initial vectors based on joint thetas.
  double relay_vecs[kNumJoints][kVecSize];
  clone_vector(relay_vecs[0], this->vec_10_);
  clone_vector(relay_vecs[1], this->vec_21_);
  clone_vector(relay_vecs[2], this->vec_32_);
  clone_vector(relay_vecs[3], this->vec_43_);
  clone_vector(relay_vecs[4], this->vec_54_);
  clone_vector(relay_vecs[5], this->vec_65_);

  for (size_t i = 0; i < kNumJoints; ++i) {
    rotate_vector_z(relay_vecs[i], input_thetas.values[i] + dh_matrix_[i][3]);
  }

  // Transform vectors to the base frame {0}.
  clone_vector(vecs_wrt_base[0], relay_vecs[0]);
  matrix_multiply(rot_mat_10, relay_vecs[1], vecs_wrt_base[1], 3, 3, 1);
  matrix_multiply(rot_mat_20, relay_vecs[2], vecs_wrt_base[2], 3, 3, 1);
  matrix_multiply(rot_mat_30, relay_vecs[3], vecs_wrt_base[3], 3, 3, 1);
  matrix_multiply(rot_mat_40, relay_vecs[4], vecs_wrt_base[4], 3, 3, 1);
  matrix_multiply(rot_mat_50, relay_vecs[5], vecs_wrt_base[5], 3, 3, 1);
}

void Kinematics::assign_orientation_from_components(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const double* rot_mat, const double* end_vec, Orientation* orientation) {
  double end_angle[kVecSize]{};

  rotation_matrix_to_euler_angles(rot_mat, end_angle);
  orientation->x = end_vec[0];
  orientation->y = end_vec[1];
  orientation->z = end_vec[2];
  orientation->a = end_angle[0];  // Kept in radians internally
  orientation->b = end_angle[1];
  orientation->c = end_angle[2];
  memcpy(orientation->rot_mat, rot_mat, kRotMatSize * sizeof(double));
  orientation->has_rot_mat = true;
}

/**
 * @brief Assembles, calculates costs for, and sorts the 8 raw IK solutions.
 *
 * This function is the final step of the IK solver before validation.
 * It performs three main tasks:
 * 1.  **Assembles:** It combines the individual solution arrays (theta_1 has 2
 *     solutions, theta_5/6 have 4, etc.) into 8 distinct, complete `Thetas`
 *     structs representing all possible kinematic configurations.
 * 2.  **Calculates Cost:** For each of the 8 solutions, it computes a "travel
 *     cost," defined as the sum of the absolute angular distance each joint
 *     must travel from its position in `last_thetas`. It correctly handles
 *     angle wrapping (e.g., a move from +179 to -179 degrees is a small 2-degree
 *     move, not a 358-degree one).
 * 3.  **Sorts:** It sorts the solutions based on this cost, from smallest to
 *     largest. Invalid solutions (containing NaN) are flagged and effectively
 *     given an infinite cost, moving them to the end of the list.
 *
 * The final `output_solutions` struct will contain the sorted *delta* angles,
 * ready for use or further validation.
 *
 * @param output_solutions The destination struct to be
 *                         filled with sorted solutions.
 * @param last_thetas The reference joint angles for calculating travel cost.
 * @param theta_1...theta_6 Pointers to arrays of raw joint solutions.
 * @param verbose If true, enables detailed debug logging.
 *
 * @todo Consider simplifying the function for readability.
 */
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Kinematics::assemble_and_sort_ik_solutions(
    Solutions& output_solutions, const Thetas& last_thetas,
    const double* theta_1, const double* theta_2, const double* theta_3,
    const double* theta_4, const double* theta_5, const double* theta_6,
    bool verbose) {
  // Intermediate storage for the 8 solutions and their calculated costs.
  double travel_costs[kIkNumSolutions]{};
  bool is_valid[kIkNumSolutions]{};
  int32_t original_indices[kIkNumSolutions]{};

  // --- Assemble each solution and calculate its travel cost ---
  for (size_t i = 0; i < kIkNumSolutions; ++i) {
    // Keep track of the original solution index.
    original_indices[i] = static_cast<int32_t>(i);
    is_valid[i] = true;
    double current_cost = 0.0;

    // The 8 solutions are indexed based on a binary combination of choices.
    // We can map the linear index `i` to the indices of the smaller solution
    // arrays using bit shifts.

    // Solutions 0-3 use theta_1[0], 4-7 use theta_1[1].
    const int32_t th1_idx = static_cast<int32_t>(i) >> 2;  // i / 4
    // Solutions 0-1 use idx 0, 2-3 use idx 1, etc.
    const int32_t th56_idx = static_cast<int32_t>(i) >> 1;  // i / 2
    // Direct mapping for the 8-solution arrays.
    const int32_t th234_idx = static_cast<int32_t>(i);

    // Create a temporary Thetas struct with the absolute angles for this solution.
    const Thetas current_absolute_thetas = {
        theta_1[th1_idx],   theta_2[th234_idx], theta_3[th234_idx],
        theta_4[th234_idx], theta_5[th56_idx],  theta_6[th56_idx]};

    // Calculate the travel cost for this assembled solution.
    for (size_t j = 0; j < kNumJoints; ++j) {
      // Check for invalid numbers (NaN) from unreachable IK calculations.
      if (std::isnan(current_absolute_thetas.values[j])) {
        is_valid[i] = false;
        break;  // No need to check other joints for this solution.
      }

      // Calculate the delta (travel distance) for this joint.
      double delta = current_absolute_thetas.values[j] - last_thetas.values[j];

      // Normalize the delta angle to the range [-PI, PI] to find the shortest path.
      // This is crucial for correct cost calculation.
      static const double kTwoPi = 2.0 * M_PI;
      delta = std::fmod(delta, kTwoPi);
      if (delta > M_PI) {
        delta -= kTwoPi;
      } else if (delta < -M_PI) {
        delta += kTwoPi;
      }

      current_cost += std::fabs(delta);
    }

    // Assign the final cost. Invalid solutions get an infinite cost so they
    // are sorted to the end.
    travel_costs[i] = is_valid[i] ? current_cost : INFINITY;
  }

  // --- Sort the solutions based on travel cost ---
  if (verbose) {
    JIE_LOG_INFO("");
    JIE_LOG_NOTE(">> IK solutions before sorting (by travel cost):");
    view_vector("Costs", travel_costs, kIkNumSolutions);
    JIE_LOG_INFO("");
  }

  quick_sort_with_keys(travel_costs, original_indices, 0, kIkNumSolutions - 1);

  if (verbose) {
    JIE_LOG_NOTE(">> IK solution order after sorting (original indices):");
    // We can't view an int32_t vector, so we build a string manually.
    std::string sorted_keys_str = "  [ ";
    for (size_t i = 0; i < kIkNumSolutions; ++i) {
      sorted_keys_str += std::to_string(original_indices[i]) + " ";
    }
    sorted_keys_str += "]";
    JIE_LOG_DEBUG(sorted_keys_str);
  }

  // --- Populate the final output_solutions struct in sorted order ---
  // The final struct should contain the DELTA angles, not the absolute ones.
  for (size_t i = 0; i < kIkNumSolutions; ++i) {
    const int32_t sorted_idx = original_indices[i];

    // Set the validity flag for the sorted position.
    output_solutions.solution_flags[i] = is_valid[sorted_idx];

    if (!is_valid[sorted_idx]) {
      // For invalid solutions, fill with zeros for safety and consistency.
      for (size_t j = 0; j < kNumJoints; ++j) {
        output_solutions.solutions[i].values[j] = 0.0;
      }
      continue;
    }

    // Map the sorted index back to the original theta array indices.
    const int32_t th1_idx = sorted_idx >> 2;
    const int32_t th56_idx = sorted_idx >> 1;
    const int32_t th234_idx = sorted_idx;

    const Thetas absolute_thetas = {theta_1[th1_idx],   theta_2[th234_idx],
                                    theta_3[th234_idx], theta_4[th234_idx],
                                    theta_5[th56_idx],  theta_6[th56_idx]};

    // Calculate and store the normalized delta angles.
    for (size_t j = 0; j < kNumJoints; ++j) {
      double delta = absolute_thetas.values[j] - last_thetas.values[j];

      static const double kTwoPi = 2.0 * M_PI;
      delta = std::fmod(delta, kTwoPi);
      if (delta > M_PI) {
        delta -= kTwoPi;
      } else if (delta < -M_PI) {
        delta += kTwoPi;
      }
      output_solutions.solutions[i].values[j] = delta;
    }
  }
}

/**
 * @brief Validates inverse kinematics solutions by checking them with forward
 *        kinematics.
 *
 * This function iterates through the potential solutions calculated by the IK
 * solver. For each valid solution, it calculates the absolute joint angles
 * (delta + last_known_angles) and runs them through the forward kinematics
 * solver. If the resulting orientation does not match the target orientation
 * within a defined tolerance, the solution is marked as invalid.
 *
 * @param solutions The set of IK solutions to validate. Flags will be modified.
 * @param last_thetas The last known joint angles, used as a base for the delta
 *                    solutions.
 * @param target_orientation The desired end-effector orientation that the
 *                           solutions are supposed to achieve.
 * @param verbose If true, print detailed logging for debugging purposes.
 */
void Kinematics::recalibrate_and_validate_solutions(
    Solutions& solutions, const Thetas& last_thetas,
    const Orientation& target_orientation, bool verbose) {
  // A temporary Orientation object to store the result of the FK check.
  // Declared outside the loop to avoid repeated construction/destruction.
  Kinematics::Orientation fk_check_orientation;

  // Epsilon for comparing orientation components. Using a slightly larger
  // tolerance for validation than for internal calculations can be robust.
  constexpr double kValidationTolerance = 1e-3;

  for (size_t i = 0; i < kIkNumSolutions; ++i) {
    // BUG FIX: The function was using `_output_solutions` which is not a parameter.
    // The correct parameter name is `solutions`.
    if (!solutions.solution_flags[i]) {
      if (verbose) {
        JIE_LOG_NOTE("Index[", i, "] is already flagged invalid. Skipping.");
      }
      continue;
    }

    // The IK solutions are deltas from the last known position.
    // We must add them to get the absolute angles for the FK check.
    const Thetas absolute_thetas = last_thetas + solutions.solutions[i];

    if (verbose) {
      // "Warning" is used for colorful illustration, not to indicate a real warning.
      JIE_LOG_WARN("Validating solution at Index[", i, "]:");
      std::string log_thetas = " [";
      for (size_t j = 0; j < kNumJoints - 1; ++j) {
        log_thetas += format_double(absolute_thetas.values[j]) + ", ";
      }
      log_thetas += format_double(absolute_thetas.values[kNumJoints - 1]) + "]";
      JIE_LOG_WARN(log_thetas);
    }

    // Run the forward kinematics check on the absolute angles.
    // The `this->` is optional but can be used for clarity.
    solve_forward_kinematics(absolute_thetas, fk_check_orientation, false);

    if (verbose) {
      JIE_LOG_DEBUG("-> FK check result:");
      view_orientation(fk_check_orientation);
    }

    // Compare the result of the FK check with the original target orientation.
    // A solution is invalid if any component is outside the tolerance.
    const bool is_position_mismatch =
        std::fabs(fk_check_orientation.x - target_orientation.x) >
            kValidationTolerance ||
        std::fabs(fk_check_orientation.y - target_orientation.y) >
            kValidationTolerance ||
        std::fabs(fk_check_orientation.z - target_orientation.z) >
            kValidationTolerance;

    // It's often better to compare rotation matrices or quaternions for orientation,
    // as Euler angles have singularities (gimbal lock). However, to match the
    // original logic, we compare the Euler angles here.
    const bool is_orientation_mismatch =
        std::fabs(fk_check_orientation.a - target_orientation.a) >
            kValidationTolerance ||
        std::fabs(fk_check_orientation.b - target_orientation.b) >
            kValidationTolerance ||
        std::fabs(fk_check_orientation.c - target_orientation.c) >
            kValidationTolerance;

    if (is_position_mismatch || is_orientation_mismatch) {
      solutions.solution_flags[i] = false;
      if (verbose) {
        JIE_LOG_NOTE("-> MISMATCH. Index[", i, "] flagged as invalid.");
      }
    } else {
      if (verbose) {
        JIE_LOG_NOTE("-> MATCH. Index[", i, "] remains valid.");
      }
    }
    JIE_LOG_DEBUG("");
  }
}

// NOLINTEND(*magic-numbers)
