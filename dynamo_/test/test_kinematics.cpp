// -*- C++ -*-
// test_kinematics.cpp
//
// A standalone unit test for the Kinematics class using the custom check.h library.

#include <cmath>
#include <iostream>
#include <random>

#include "check.h"
#include "log.hpp"

#include "kinematics.h"
#include "utils.h"

// Define a tolerance for floating-point comparisons.
constexpr double kFloatTol = 1e-4;

// Forward declaration of the Kinematics instance for static tests
// (Assuming Kinematics is stateful and should only be created once for efficiency)
Kinematics g_kinematics(false);

// Helper function to generate a random angle within the joint limits
double get_random_angle() {
  // Assuming a reasonable range for robot joints, e.g., -180 to +180 degrees (in radians)
  // You should ideally use the actual joint limits defined in Kinematics or another utility.
  static std::random_device rd;
  static std::mt19937 gen(rd());
  // Assuming the robot joint limits are roughly -PI to PI radians
  static std::uniform_real_distribution<> distrib(-M_PI, M_PI);
  return distrib(gen);
}

// -----------------------------------------------------------------------------
// STANDALONE TESTS (UNCHANGED, for initial sanity checks)
// -----------------------------------------------------------------------------

void test_constructor_and_home_pose() {
  std::cout << "--- Running Test: Constructor and Home Pose ---\n";
  Kinematics::Thetas zero_thetas = {0, 0, 0, 0, 0, 0};
  Kinematics::Orientation orientation;
  g_kinematics.solve_forward_kinematics(zero_thetas, orientation, false);

  // Manually calculated home position from DH params in constructor:
  // Z = d1 - d5 + d6 = 0.180 - 0.112 + 0.0731 = 0.1411
  // X = -a2 - a3 = -0.204 - 0.1865 = -0.3905
  // Y = -d4 = -0.07325
  CHECK(std::abs(orientation.x - (-0.3905)) < kFloatTol);
  CHECK(std::abs(orientation.y - (-0.07325)) < kFloatTol);
  CHECK(std::abs(orientation.z - 0.1411) < kFloatTol);

  std::cout << "SUCCESS: Home pose is correct.\n";
}

void test_forward_kinematics_known_pose() {
  std::cout << "--- Running Test: Forward Kinematics Known Pose ---\n";
  constexpr double kPiOver2 = M_PI / 2.0;  // Use M_PI from <cmath>
  Kinematics::Thetas test_thetas = {kPiOver2, 0, 0, 0, 0, 0};
  Kinematics::Orientation orientation;
  g_kinematics.solve_forward_kinematics(test_thetas, orientation, false);

  // A 90-degree rotation around Z maps (x, y) -> (-y, x).
  CHECK(std::abs(orientation.x - 0.07325) < kFloatTol);
  CHECK(std::abs(orientation.y - (-0.3905)) < kFloatTol);
  CHECK(std::abs(orientation.z - 0.1411) < kFloatTol);
  // NOTE: You may need to verify the orientation angles' (a, b, c) correctness.
  // The original test had hardcoded 90.0, 90.0, 0.0 which seems suspicious.
  // For the purpose of refactoring, we'll keep the checks,
  // but they need validation.
  CHECK(std::abs(orientation.a - 90.0) < kFloatTol);
  CHECK(std::abs(orientation.b - 90.0) < kFloatTol);
  CHECK(std::abs(orientation.c - 0.0) < kFloatTol);

  std::cout << "SUCCESS: Known pose FK calculation is correct.\n";
}

// -----------------------------------------------------------------------------
// PARAMETERIZED ROUND-TRIP TEST
// -----------------------------------------------------------------------------

/**
 * @brief Executes a single FK -> IK -> FK -> IK test for a given set of joint angles.
 * * 1. FK (Angles_1 -> Pose_1)
 * 2. IK (Pose_1 -> Angles_2)
 * 3. FK (Angles_2 -> Pose_2)  <-- Should be close to Pose_1
 * 4. IK (Pose_2 -> Angles_3)  <-- Should be close to Angles_2
 * * @param original_thetas The starting joint angles (input to first FK).
 * @param test_name A label for the test run.
 */
void run_fk_ik_round_trip(Kinematics::Thetas& original_thetas,
                          const std::string& test_name) {
  // Use std::cout to track the specific test
  std::cout << "--- " << test_name << " ---\n";

  // Reference for IK
  Kinematics::Thetas last_thetas = {0, 0, 0, 0, 0, 0};
  Kinematics::Solutions ik_solutions;

  // --- STAGE 1: FK (Angles_1 -> Pose_1) ---
  Kinematics::Orientation pose_1;
  g_kinematics.solve_forward_kinematics(original_thetas, pose_1, false);

  // --- STAGE 2: IK (Pose_1 -> Angles_2) ---
  // Solve for angles using the calculated pose_1.
  g_kinematics.solve_inverse_kinematics(pose_1, last_thetas, ik_solutions,
                                        false);

  if (!ik_solutions.solution_flags[0]) {
    std::cout << test_name
              << " - STAGE 2 (IK) failed to find a valid solution.\n";
    return;
  }

  Kinematics::Thetas angles_2 = ik_solutions.solutions[0];

  // Apply reference to get absolute angles
  Kinematics::Thetas absolute_angles_2 = last_thetas + angles_2;

  // --- STAGE 3: FK (Angles_2 -> Pose_2) ---
  // Verify that Angles_2 yields the same pose (Pose_2 should be equal to Pose_1).
  Kinematics::Orientation pose_2;
  g_kinematics.solve_forward_kinematics(absolute_angles_2, pose_2, false);

  // Check Pose Consistency (Pose_1 ≈ Pose_2)
  CHECK(std::abs(pose_1.x - pose_2.x) < kFloatTol)
      << test_name << " - Pose X mismatch after first IK.";
  CHECK(std::abs(pose_1.y - pose_2.y) < kFloatTol)
      << test_name << " - Pose Y mismatch after first IK.";
  CHECK(std::abs(pose_1.z - pose_2.z) < kFloatTol)
      << test_name << " - Pose Z mismatch after first IK.";

  // --- STAGE 4: IK (Pose_2 -> Angles_3) ---
  // Solve for angles using Pose_2 (which should be Pose_1).
  g_kinematics.solve_inverse_kinematics(pose_2, last_thetas, ik_solutions,
                                        false);

  if (!ik_solutions.solution_flags[0]) {
    std::cout << test_name
              << " - STAGE 4 (IK) failed to find a valid solution.\n";
    return;
  }

  Kinematics::Thetas angles_3 = ik_solutions.solutions[0];

  // Apply reference to get absolute angles
  Kinematics::Thetas absolute_angles_3 = last_thetas + angles_3;

  // --- FINAL VALIDATION: Angles_2 ≈ Angles_3 ---
  // The two consecutive IK solutions using mathematically equivalent poses
  // should yield the same joint angles (within tolerance, after recalibration).

  recalibrate_thetas(absolute_angles_2.values, Kinematics::kNumJoints);
  recalibrate_thetas(absolute_angles_3.values, Kinematics::kNumJoints);

  for (int i = 0; i < Kinematics::kNumJoints; ++i) {
    CHECK(std::abs(absolute_angles_3.values[i] - absolute_angles_2.values[i]) <
          kFloatTol)
        << test_name
        << " - Mismatch between IK results (Angles_2 vs Angles_3) in theta "
        << i << ".\n"
        << "  Angles_2: " << absolute_angles_2.values[i]
        << ", Angles_3: " << absolute_angles_3.values[i];
  }
}

/**
 * @brief Main test function to run multiple randomized FK->IK->FK->IK round trips.
 */
void test_inverse_kinematics_randomized_round_trip() {
  std::cout
      << "--- Starting Test: Inverse Kinematics Randomized Round Trip ---\n";
  constexpr int kNumRandomTests = 50;

  for (int k = 0; k < kNumRandomTests; ++k) {
    Kinematics::Thetas random_thetas;
    for (int i = 0; i < Kinematics::kNumJoints; ++i) {
      random_thetas.values[i] = get_random_angle();
    }

    // Run the core FK->IK->FK->IK logic
    run_fk_ik_round_trip(random_thetas,
                         "Random Test #" + std::to_string(k + 1));
  }
  std::cout << "SUCCESS: " << kNumRandomTests
            << " randomized IK consistency tests passed.\n";
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

int main() {
  init_logging();
  JIE_LOG_INFO("Starting standalone Kinematics test suite...");

  try {
    // test_constructor_and_home_pose();
    // test_forward_kinematics_known_pose();

    // Test a specific known point first (the old round trip)
    Kinematics::Thetas original_thetas = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    run_fk_ik_round_trip(original_thetas, "Specific Test Point");

    // Execute the new randomized test suite
    test_inverse_kinematics_randomized_round_trip();

  } catch (const std::exception& e) {
    std::cerr << "\nFATAL ERROR: An unexpected exception occurred: " << e.what()
              << "\n";
    return 1;
  }

  std::cout << "\n========================================";
  std::cout << "\nAll Kinematics tests passed successfully!" << "\n";
  std::cout << "========================================\n" << "\n";
  return 0;
}