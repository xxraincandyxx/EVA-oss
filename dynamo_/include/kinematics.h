// -*- C++ -*-
// kinematics.h

#ifndef EVA_KINEMATICS_H_
#define EVA_KINEMATICS_H_

#include <cstdint>
#include <string>

// Forward declare logging helpers to reduce header dependencies.
void view_vector(const std::string& name, const double* vec,
                 int32_t length = 3);
void view_matrix(const std::string& name, const double* mat, int32_t m = 3,
                 int32_t n = 3);

class Kinematics {
 public:
  // --- Public Data Structures ---
  static constexpr int32_t kNumJoints = 6;
  static constexpr int32_t kNumSolutions = 8;
  static constexpr int32_t kRotMatSize = 9;
  static constexpr double kRadToDeg = 57.295777754771045;
  static constexpr double kDegToRad = 1.0 / kRadToDeg;

  struct Thetas {
    Thetas() = default;

    Thetas(double th1, double th2, double th3, double th4, double th5,
           double th6)
        : values{th1, th2, th3, th4, th5, th6} {}

    double values[kNumJoints]{};

    friend Thetas operator+(const Thetas& lhs, const Thetas& rhs) {
      Thetas result{};
      for (size_t i = 0; i < kNumJoints; ++i) {
        result.values[i] = lhs.values[i] + rhs.values[i];
      }
      return result;
    }

    friend Thetas operator-(const Thetas& lhs, const Thetas& rhs) {
      Thetas result{};
      for (size_t i = 0; i < kNumJoints; ++i) {
        result.values[i] = lhs.values[i] - rhs.values[i];
      }
      return result;
    }
  };

  struct Orientation {
    Orientation() = default;

    Orientation(double x, double y, double z, double a, double b, double c)
        : x(x), y(y), z(z), a(a), b(b), c(c) {}

    double x = 0.0, y = 0.0, z = 0.0;  // Position in meters.
    double a = 0.0, b = 0.0, c = 0.0;  // Orientation (Euler angles) in degrees.
    double rot_mat[kRotMatSize]{};     // 3x3 Rotation Matrix.
    bool has_rot_mat{false};
  };

  struct Solutions {
    Thetas solutions[kNumSolutions]{};
    bool solution_flags[kNumSolutions]{};
  };

  // --- API ---
  explicit Kinematics(bool enable_cache = false);
  virtual ~Kinematics();

  bool solve_forward_kinematics(const Thetas& input_thetas,
                                Orientation& output_orientation,
                                bool verbose = false);
  bool solve_inverse_kinematics(const Orientation& input_orientation,
                                const Thetas& last_thetas,
                                Solutions& output_solutions,
                                bool verbose = false);
  void get_states(double** states, const Thetas& input_thetas,
                  bool verbose = false);

  void view_dh_matrix();
  Orientation* get_cached_orientations() const;

 private:
  // --- Private Constants ---
  static constexpr double kEpsilon = 1e-5;
  static constexpr int32_t kDhParams = 4;
  static constexpr int32_t kVecSize = 3;

  // --- Private Member Variables ---
  double d1_, a2_, a3_, d4_, d5_, d6_;
  double dh_matrix_[kNumJoints][kDhParams]{};
  double vec_10_[kVecSize]{}, vec_21_[kVecSize]{}, vec_32_[kVecSize]{},
      vec_43_[kVecSize]{}, vec_54_[kVecSize]{}, vec_65_[kVecSize]{};
  double pos_01_[kVecSize]{}, pos_12_[kVecSize]{}, pos_23_[kVecSize]{},
      pos_34_[kVecSize]{}, pos_45_[kVecSize]{}, pos_56_[kVecSize]{};

  bool enable_cache_;
  Orientation* cached_orientations_ = nullptr;

  // --- Private Helper Methods ---
  void solve_ik_theta1(double* theta_1, const double* pos_60,
                       const double* rot_mat_60, bool verbose) const;
  void solve_ik_theta5(double* theta_5, double* pos_61, const double* pos_60,
                       const double* rot_mat_01, bool verbose) const;
  void solve_ik_theta6(double& theta_6, const double* rot_mat_60,
                       const double* rot_mat_01, double theta_5,
                       bool verbose) const;
  void solve_ik_theta3(double* theta_3, double* rot_mat_45, double* pos_31,
                       const double* rot_mat_56, const double* rot_mat_01,
                       const double* rot_mat_60, const double* pos_61,
                       double theta_5, double theta_6, bool verbose) const;
  void solve_ik_theta2(double* theta_2, const double* pos_31,
                       const double* theta_3, bool verbose) const;
  void solve_ik_theta4(double& theta_4, const double* rot_mat_45,
                       const double* rot_mat_56, const double* rot_mat_60,
                       const double* rot_mat_01, const double& theta_2,
                       const double& theta_3) const;
  void compute_fk_transforms(const Thetas& input_thetas,
                             double (*rot_mats)[kRotMatSize],
                             double (*vecs_wrt_base)[kVecSize],
                             double* rot_mat_60, bool verbose) const;
  static void assign_orientation_from_components(const double* rot_mat,
                                                 const double* end_vec,
                                                 Orientation* orientation);
  static void assemble_and_sort_ik_solutions(
      Solutions& output_solutions, const Thetas& last_thetas,
      const double* theta_1, const double* theta_2, const double* theta_3,
      const double* theta_4, const double* theta_5, const double* theta_6,
      bool verbose);

  void recalibrate_and_validate_solutions(Solutions& solutions,
                                          const Thetas& last_thetas,
                                          const Orientation& target_orientation,
                                          bool verbose);
};

#endif  // EVA_KINEMATICS_H_
