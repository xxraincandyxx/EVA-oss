// -*- C++ -*-
// utils.h

#ifndef EVA_UTILS_H_
#define EVA_UTILS_H_

#include <cstdint>
#include <string>

#include "kinematics.h"

// Defined in tools.cpp
void init_logging(const std::string& subdir = "");
std::string find_usb_device();

// Defined in utils.cpp (formerly functional.cpp)
std::string format_double(double value);
std::string hex_array_to_string(const uint8_t* data, size_t size);
void view_orientation(const Kinematics::Orientation& orientation);
void quick_sort_with_keys(double* arr, int32_t* keys, int32_t low,
                          int32_t high);
void recalibrate_thetas(double* arr, int32_t size);
double vector_norm(const double* vector, bool return_square = false);
void matrix_multiply(const double* matrix1, const double* matrix2,
                     double* matrix_out, int32_t m, int32_t l, int32_t n);
void theta_to_rotation_matrix(double theta, double* rotation_m);
void rotation_matrix_to_euler_angles(const double* rotation_m,
                                     double* euler_angles);
void euler_angles_to_rotation_matrix(const double* euler_angles,
                                     double* rotation_m);
void orientation_to_direction_vector(double a, double b, double c, double& dx,
                                     double& dy, double& dz);
void direction_vector_to_orientation(double dx, double dy, double dz, double& a,
                                     double& b, double& c);

#endif  // EVA_UTILS_H_
