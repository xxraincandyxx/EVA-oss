// -*- C++ -*-
// utils.cpp
//
// Implements general mathematical and logging helper functions.

#include "utils.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "log.hpp"

namespace {
constexpr double kEpsilon = 1e-8;
constexpr int32_t kDefaultDoubleWidth = 6;
constexpr int32_t kDefaultDoublePrecision = 3;
constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfPi = 1.57079632679489661923;

// Helper for quick_sort_with_keys.
int32_t partition(double* arr, int32_t* keys, int32_t low, int32_t high) {
  const double pivot = arr[high];
  int32_t i = low - 1;
  for (int32_t j = low; j < high; ++j) {
    if (arr[j] <= pivot) {
      i++;
      std::swap(arr[i], arr[j]);
      std::swap(keys[i], keys[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  std::swap(keys[i + 1], keys[high]);
  return i + 1;
}

}  // namespace

// --- Logging Helper Functions ---

std::string format_double(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(kDefaultDoublePrecision)
      << std::setw(kDefaultDoubleWidth) << value;
  return oss.str();
}

std::string hex_array_to_string(const uint8_t* data, size_t size) {
  std::ostringstream oss;
  oss << "0x";
  for (size_t i = 0; i < size; ++i) {
    oss << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int32_t>(data[i]);
  }
  return oss.str();
}

void view_vector(const std::string& name, const double* vec, int32_t length) {
  std::string log_vec = "  [";
  if (length > 0) {
    for (int32_t i = 0; i < length - 1; ++i) {
      log_vec += format_double(vec[i]) + " ";
    }
    log_vec += format_double(vec[length - 1]);
  }
  log_vec += "]";
  JIE_LOG_DEBUG(name, ": ", log_vec);
}

void view_matrix(const std::string& name, const double* mat, int32_t m,
                 int32_t n) {
  JIE_LOG_INFO(name, ":");
  for (int32_t i = 0; i < m; ++i) {
    std::string log_row;
    for (int32_t j = 0; j < n; ++j) {
      log_row += " " + format_double(mat[i * n + j]);
    }
    JIE_LOG_DEBUG(log_row);
  }
}

void view_orientation(const Kinematics::Orientation& orientation) {
  JIE_LOG_DEBUG(" X:", format_double(orientation.x),
                " Y:", format_double(orientation.y),
                " Z:", format_double(orientation.z));
  JIE_LOG_DEBUG(" A:", format_double(orientation.a),
                " B:", format_double(orientation.b),
                " C:", format_double(orientation.c));
}

// --- General Mathematical Helpers ---

void quick_sort_with_keys(double* arr, int32_t* keys, int32_t low,
                          int32_t high) {
  if (low < high) {
    int32_t pivot_index = partition(arr, keys, low, high);
    quick_sort_with_keys(arr, keys, low, pivot_index - 1);
    quick_sort_with_keys(arr, keys, pivot_index + 1, high);
  }
}

void recalibrate_thetas(double* arr, int32_t size) {
  static const double kTwoPi = 2.0 * kPi;
  for (int32_t i = 0; i < size; ++i) {
    arr[i] = std::fmod(arr[i], kTwoPi);
    if (arr[i] > kPi) {
      arr[i] -= kTwoPi;
    } else if (arr[i] < -kPi) {
      arr[i] += kTwoPi;
    }
  }
}

double vector_norm(const double* vector, bool return_square) {
  double sum_sq = (vector[0] * vector[0]) + (vector[1] * vector[1]) +
                  (vector[2] * vector[2]);
  return return_square ? sum_sq : std::sqrt(sum_sq);
}

void matrix_multiply(const double* matrix1, const double* matrix2,
                     double* matrix_out, int32_t m, int32_t l, int32_t n) {
  for (int32_t i = 0; i < m; ++i) {
    for (int32_t j = 0; j < n; ++j) {
      double tmp = 0.0;
      for (int32_t k = 0; k < l; ++k) {
        tmp += matrix1[i * l + k] * matrix2[k * n + j];
      }
      matrix_out[i * n + j] = tmp;
    }
  }
}

void theta_to_rotation_matrix(const double theta, double* rotation_matrix) {
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  rotation_matrix[0] = cos_theta;
  rotation_matrix[1] = -sin_theta;
  rotation_matrix[2] = 0.0;
  rotation_matrix[3] = sin_theta;
  rotation_matrix[4] = cos_theta;
  rotation_matrix[5] = 0.0;
  rotation_matrix[6] = 0.0;
  rotation_matrix[7] = 0.0;
  rotation_matrix[8] = 1.0;
}

void rotation_matrix_to_euler_angles(const double* rotation_matrix,
                                     double* euler_angles) {
  double a{}, b{}, c{};
  if (std::fabs(rotation_matrix[6]) >= 1.0 - kEpsilon) {  // Gimbal lock
    if (rotation_matrix[6] < 0) {                         // Pointing up
      a = 0.0;
      b = kHalfPi;
      c = std::atan2(rotation_matrix[1], rotation_matrix[4]);
    } else {  // Pointing down
      a = 0.0;
      b = -kHalfPi;
      c = -std::atan2(rotation_matrix[1], rotation_matrix[4]);
    }
  } else {
    b = std::atan2(-rotation_matrix[6],
                   std::sqrt(rotation_matrix[0] * rotation_matrix[0] +
                             rotation_matrix[3] * rotation_matrix[3]));
    const double cos_b = std::cos(b);
    a = std::atan2(rotation_matrix[3] / cos_b, rotation_matrix[0] / cos_b);
    c = std::atan2(rotation_matrix[7] / cos_b, rotation_matrix[8] / cos_b);
  }
  euler_angles[0] = c;  // Yaw
  euler_angles[1] = b;  // Pitch
  euler_angles[2] = a;  // Roll
}

void euler_angles_to_rotation_matrix(const double* euler_angles,
                                     double* rotation_matrix) {
  // ZYX convention: R = Rz(c) * Ry(b) * Rx(a)
  const double cc = std::cos(euler_angles[0]);
  const double sc = std::sin(euler_angles[0]);
  const double cb = std::cos(euler_angles[1]);
  const double sb = std::sin(euler_angles[1]);
  const double ca = std::cos(euler_angles[2]);
  const double sa = std::sin(euler_angles[2]);

  rotation_matrix[0] = ca * cb;
  rotation_matrix[1] = ca * sb * sc - sa * cc;
  rotation_matrix[2] = ca * sb * cc + sa * sc;
  rotation_matrix[3] = sa * cb;
  rotation_matrix[4] = sa * sb * sc + ca * cc;
  rotation_matrix[5] = sa * sb * cc - ca * sc;
  rotation_matrix[6] = -sb;
  rotation_matrix[7] = cb * sc;
  rotation_matrix[8] = cb * cc;
}

void orientation_to_direction_vector(const double a, const double b,
                                     const double c, double& dx, double& dy,
                                     double& dz) {
  const double ca = std::cos(a);
  const double sa = std::sin(a);
  const double cb = std::cos(b);
  const double sb = std::sin(b);
  const double cc = std::cos(c);
  const double sc = std::sin(c);

  dx = cc * sb * ca + sc * sa;
  dy = sc * sb * ca - cc * sa;
  dz = cb * ca;

  const double length = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (length > kEpsilon) {
    dx /= length;
    dy /= length;
    dz /= length;
  }
}

void direction_vector_to_orientation(double dx, double dy, double dz, double& a,
                                     double& b, double& c) {
  double length = std::sqrt(dx * dx + dy * dy + dz * dz);
  if (length < kEpsilon) {
    dx = 0.0;
    dy = 0.0;
    dz = 1.0;
    length = 1.0;
  }
  dx /= length;
  dy /= length;
  dz /= length;

  a = 0.0;  // Assume A=0 since direction vector only has 2 DOF
  b = std::acos(dz);
  if (std::isnan(b))
    b = 0.0;

  if (std::fabs(std::sin(b)) < kEpsilon) {
    c = 0.0;
  } else {
    c = std::atan2(dy, dx);
  }
}
