// -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil; -*-
// bindings.cpp

#include <pybind11/detail/common.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "armctrl.h"
#include "dynamo.hpp"

namespace py = pybind11;

using eva::ctrl::ArmCtrl;
using eva::ctrl::InstanceStreamer;
using eva::ctrl::PumpCtrl;
using eva::ctrl::RotCtrl;

PYBIND11_MODULE(dynamo_, m) {
  m.doc() = "Python bindings for dynamo_";

  m.def("init_logging", &init_logging,
        "Initialize files and sinks for dynamo_");

  py::class_<Kinematics::Thetas>(m, "Thetas")
      .def(py::init())
      .def(py::init<double, double, double, double, double, double>())
      .def_property(
          "thetas",
          [](const Kinematics::Thetas& T) {
            py::list lst;
            for (int i = 0; i < 6; i++)
              lst.append(T.values[i]);
            return lst;
          },
          [](Kinematics::Thetas& T, py::sequence seq) {
            if (seq.size() != 6)
              throw py::value_error("`thetas` must have absolutely 6 elements");
            for (int i = 0; i < 6; i++)
              T.values[i] = seq[i].cast<double>();
          })
      .def(py::self + py::self)
      .def(py::self - py::self);

  py::class_<Kinematics::Orientation>(m, "Orientation")
      .def(py::init<>())
      .def(py::init<double, double, double, double, double, double>(),
           py::arg("x"), py::arg("y"), py::arg("z"), py::arg("a"), py::arg("b"),
           py::arg("c"))
      .def_readwrite("x", &Kinematics::Orientation::x)
      .def_readwrite("y", &Kinematics::Orientation::y)
      .def_readwrite("z", &Kinematics::Orientation::z)
      .def_readwrite("a", &Kinematics::Orientation::a)
      .def_readwrite("b", &Kinematics::Orientation::b)
      .def_readwrite("c", &Kinematics::Orientation::c)
      .def_readwrite("has_rot_mat", &Kinematics::Orientation::has_rot_mat)
      .def_property(
          "RotMat",
          [](const Kinematics::Orientation& O) {
            py::list lst;
            for (int i = 0; i < 9; i++)
              lst.append(O.rot_mat[i]);
            return lst;
          },
          [](Kinematics::Orientation& O, py::sequence seq) {
            if (seq.size() != 9)
              throw py::value_error("`RotMat` must have absolutely 9 elements");
            for (int i = 0; i < 9; i++)
              O.rot_mat[i] = seq[i].cast<double>();
          });

  py::class_<Kinematics::Solutions>(m, "Solutions")
      .def(py::init<>())
      .def_property(
          "solutions",
          [](const Kinematics::Solutions& S) {
            py::list lst;
            for (int i = 0; i < 8; i++)
              lst.append(S.solutions[i]);
            return lst;
          },
          [](Kinematics::Solutions& S, py::sequence solutions) {
            if (solutions.size() != 8)
              throw py::value_error(
                  "`solutions` must have absolutely 8 elements");
            for (int i = 0; i < 8; i++)
              S.solutions[i] = solutions[i].cast<Kinematics::Thetas>();
          })
      .def_property(
          "solution_flags",
          [](const Kinematics::Solutions& S) {
            py::list lst;
            for (int i = 0; i < 9; i++)
              lst.append(S.solution_flags[i]);
            return lst;
          },
          [](Kinematics::Solutions& S, const py::sequence& solution_flags) {
            if (solution_flags.size() != 8) {
              throw py::value_error(
                  "`solution_flags` must have absolutely 8 elements");
            }
            for (int i = 0; i < 8; i++)
              S.solution_flags[i] = solution_flags[i].cast<bool>();
          });

  py::class_<Kinematics>(m, "Kinematics")
      .def(py::init<bool>(), py::arg("enable_cache") = false)
      .def(
          "get_states",
          [](Kinematics& self, const Kinematics::Thetas& input_thetas,
             bool verbose) -> py::object {
            // TODO
            throw std::runtime_error(
                "Kinematics::get_states() not fully implemented for external "
                "usage currently, please refer to "
                "InstanceStreamer::get_states().");

            double** states = (double**)malloc(6 * sizeof(double*));
            self.get_states(states, input_thetas, verbose);

            if (states != nullptr) {
              py::list state_lst;
              for (int i = 0; i < 6; ++i) {
                py::list vec;
                for (int j = 0; j < 3; j++)
                  vec.append(states[i][j]);
                state_lst.append(vec);
                delete[] states[i];
              }

              delete[] states;
              return state_lst;
            } else {
              delete[] states;
              return py::none();
            }
          },
          py::arg("input_thetas"), py::arg("verbose") = false)
      .def(
          "solve_forward_kinematics",
          [](Kinematics& self, const Kinematics::Thetas& input_thetas,
             bool verbose = false) {
            Kinematics::Orientation output_orientation;
            bool ret = self.solve_forward_kinematics(
                input_thetas, output_orientation, verbose);
            return output_orientation;
          },
          py::arg("input_thetas"), py::arg("verbose") = false)
      .def(
          "solve_inverse_kinematics",
          [](Kinematics& self, const Kinematics::Orientation input_orientation,
             const Kinematics::Thetas last_thetas, bool verbose = false) {
            Kinematics::Solutions solutions;
            bool ret = self.solve_inverse_kinematics(
                input_orientation, last_thetas, solutions, verbose);
            return solutions;
          },
          py::arg("input_orientation"), py::arg("last_thetas"),
          py::arg("verbose") = false)
      .def("get_cached_orientations", [](Kinematics& self) {
        py::list lst;
        Kinematics::Orientation* orientations = self.get_cached_orientations();
        for (int i = 0; i < 6; i++)
          lst.append(orientations[i]);
        return lst;
      });

  py::class_<ArmCtrl>(m, "ArmCtrl")
      .def(py::init<const std::string&, const unsigned int, bool, bool>(),
           py::arg("path") = "/dev/ttyAMA2",
           py::arg("sleep_time_interval") = 8000, py::arg("debug") = false,
           py::arg("enable_port") = true)
      .def("open", &ArmCtrl::open_port)
      .def("close_port", &ArmCtrl::close_port)
      .def("send", &ArmCtrl::send)
      .def("forward", &ArmCtrl::forward, py::arg("input_thetas"),
           py::arg("velocity") = 5, py::arg("verbose") = false)
      .def("is_open", &ArmCtrl::is_port_open)
      .def("run_loop", &ArmCtrl::run_loop)
      .def_property_readonly("device_path", &ArmCtrl::get_device_path)
      .def("__enter__",
           [](ArmCtrl& a) {
             a.open_port();
             return &a;
           })
      .def("__exit__", [](ArmCtrl& a, py::args) { a.close_port(); });

  py::class_<RotCtrl>(m, "RotCtrl")
      .def(py::init<const double, const double, const double, const double,
                    const std::string&, bool, bool>(),
           py::arg("x") = 0.24, py::arg("y") = 0.0, py::arg("z") = 0.24,
           py::arg("theta") = 0.0, py::arg("path") = "/dev/ttyAMA2",
           py::arg("debug") = false, py::arg("enable_port") = true)
      .def("open", &RotCtrl::open_port)
      .def("close_port", &RotCtrl::close_port)
      .def("send", &RotCtrl::send)
      .def("forward", &RotCtrl::forward, py::arg("rot_angle"),
           py::arg("verbose") = false)
      .def("is_open", &RotCtrl::is_port_open)
      .def_property_readonly("device_path", &RotCtrl::get_device_path)
      .def("__enter__",
           [](RotCtrl& R) {
             R.open_port();
             return &R;
           })
      .def("__exit__", [](RotCtrl& R, py::args) { R.close_port(); });

  py::class_<PumpCtrl>(m, "PumpCtrl")
      .def(py::init<const std::string&, bool, bool>(), py::arg("path"),
           py::arg("debug"), py::arg("enable_port") = true);

  py::class_<InstanceStreamer>(m, "InstanceStreamer")
      .def(py::init<Kinematics*, ArmCtrl*, RotCtrl*, PumpCtrl*,
                    Kinematics::Thetas, double, bool, bool>(),
           py::arg("kinematics"), py::arg("armctrl"), py::arg("rotctrl"),
           py::arg("pumpctrl"), py::arg("init_thetas"), py::arg("init_theta"),
           py::arg("verbose"), py::arg("debug"))
      .def(
          "forward_kinematics",
          [](InstanceStreamer& self, py::list thetas) {
            if (thetas.size() != 6)
              throw std::runtime_error(
                  "Python InstanceStreamer::forward_kinematics() - ERROR: "
                  "Input thetas must be of size 6.");

            Kinematics::Thetas input_thetas(thetas[0].cast<double>(),  // Axis 1
                                            thetas[1].cast<double>(),  // Axis 2
                                            thetas[2].cast<double>(),  // Axis 3
                                            thetas[3].cast<double>(),  // Axis 4
                                            thetas[4].cast<double>(),  // Axis 5
                                            thetas[5].cast<double>()   // Axis 6
            );

            Kinematics::Orientation output_orientation{};
            self.forward_kinematics(input_thetas, output_orientation);

            // assign results
            py::list orientation;  // init to-return list
            orientation.append(output_orientation.x);
            orientation.append(output_orientation.y);
            orientation.append(output_orientation.z);
            orientation.append(output_orientation.a);
            orientation.append(output_orientation.b);
            orientation.append(output_orientation.c);
            return orientation;
          },
          py::arg("thetas"))
      .def(
          "inverse_kinematics",
          [](InstanceStreamer& self, const py::list& orientation) {
            if (orientation.size() != 6) {
              throw std::runtime_error(
                  "Python InstanceStreamer::inverse_kinematics() - ERROR: "
                  "Input orientation must be of size 6.");
            }

            Kinematics::Orientation input_orientation(
                orientation[0].cast<double>(),  // x
                orientation[1].cast<double>(),  // y
                orientation[2].cast<double>(),  // z
                orientation[3].cast<double>(),  // a
                orientation[4].cast<double>(),  // b
                orientation[5].cast<double>()   // c
            );

            Kinematics::Thetas output_thetas{};
            self.inverse_kinematics(input_orientation, output_thetas);

            // assign results
            py::list thetas;  // init to-return list
            for (int i = 0; i < 6; i++)
              thetas.append(output_thetas.values[i]);
            return thetas;
          },
          py::arg("orientation"))
      .def(
          "dual_derive",
          [](InstanceStreamer& self, py::list input_orientation,
             py::list input_thetas, unsigned int u_duration, bool return_states)
              -> std::tuple<py::object, py::object, py::object> {
            /**
             *
             * @brief Binding InstanceStreamer::dualDerive()
             * @details Returned thetas is the solution of Inverse Kinematics,
             * not the current thetas value cached within the Kinematics.
             * @returns orientation, thetas, states
             */

            // Memory allocation for states
            double** states = nullptr;
            if (return_states) {
              states = (double**)malloc(6 * sizeof(double*));
              for (int i = 0; i < 6; i++) {
                states[i] = (double*)malloc(3 * sizeof(double));
              }
            }

            py::list py_ret_orientation = py::list();
            py::list py_ret_thetas = py::list();
            if (input_orientation.size() > 0) {
              // We recognize this using IK
              Kinematics::Orientation _input_orientation(  // c++ Style
                  input_orientation[0].cast<double>(),     // x
                  input_orientation[1].cast<double>(),     // y
                  input_orientation[2].cast<double>(),     // z
                  input_orientation[3].cast<double>(),     // a
                  input_orientation[4].cast<double>(),     // b
                  input_orientation[5].cast<double>()      // c
              );
              Kinematics::Thetas _return_thetas{};

              self.dual_derive(&_input_orientation, &_return_thetas, u_duration,
                               states, true, return_states);  // use orientation

              // Convert c++ style to Python style
              py_ret_orientation = input_orientation;
              for (int i = 0; i < 6; i++) {
                py_ret_thetas.append(_return_thetas.values[i]);
              }

            } else if (input_thetas.size() > 0) {
              // we recognize this using FK
              Kinematics::Thetas _input_thetas(input_thetas[0].cast<double>(),
                                               input_thetas[1].cast<double>(),
                                               input_thetas[2].cast<double>(),
                                               input_thetas[3].cast<double>(),
                                               input_thetas[4].cast<double>(),
                                               input_thetas[5].cast<double>());
              Kinematics::Orientation _return_orientation{};

              self.dual_derive(&_return_orientation, &_input_thetas, u_duration,
                               states, false, return_states);  // use thetas

              // Convert c++ style to Python style
              py_ret_thetas = input_thetas;
              py_ret_orientation.append(_return_orientation.x);
              py_ret_orientation.append(_return_orientation.y);
              py_ret_orientation.append(_return_orientation.z);
              py_ret_orientation.append(_return_orientation.a);
              py_ret_orientation.append(_return_orientation.b);
              py_ret_orientation.append(_return_orientation.c);

            } else {
              throw std::runtime_error(
                  "InstanceStreamer.dual_derive() received invalid "
                  "input_orientation or input_thetas.");
            }

            // Update states
            if (return_states && states != nullptr) {
              py::list state_lst;
              for (int i = 0; i < 6; ++i) {
                py::list vec;
                for (int j = 0; j < 3; j++)
                  vec.append(states[i][j]);
                state_lst.append(vec);
                delete[] states[i];
              }

              delete[] states;
              return std::make_tuple(py_ret_orientation, py_ret_thetas,
                                     state_lst);

            } else {
              return std::make_tuple(py_ret_orientation, py_ret_thetas,
                                     py::none());
            }
          },
          py::arg("input_orientation") = py::list(),
          py::arg("input_thetas") = py::list(), py::arg("u_duration") = 3600000,
          py::arg("return_states") = false)
      .def(
          "singular_derive",
          [](InstanceStreamer& self, double angle) -> py::none {
            /**
             *
             * @brief Binding InstanceStreamer::singularDerive()
             * @details Direct control to the robotic arm
             * @returns none
             */

            self.singular_derive(angle);
            return py::none();
          },
          py::arg("angle") = 0.0)
      .def("ctrl_rot",
           [](InstanceStreamer& self, int command_idx) -> py::none {
             /**
              *
              * @brief Binding InstanceStreamer::ctrlRot()
              * @param command_idx Use integer instead of enum { "CLAMP": 0, "RELEASE": 1, "ROTATE": 2 }
              */
             self.ctrl_rotation(command_idx);
             return py::none();
           })
      .def("ctrl_rot_fallback",
           [](InstanceStreamer& self, int command_idx) -> py::none {
             /**
              *
              * @brief Binding InstanceStreamer::ctrlRotFallback()
              * @param command_idx Use integer instead of enum { "CLAMP": 0, "RELEASE": 1, "ROTATE": 2 }
              */
             self.ctrl_rotation_fallback(command_idx);
             return py::none();
           })
      .def("ctrl_pump",
           [](InstanceStreamer& self, int command_idx) -> py::none {
             /**
              *
              * @brief Binding InstanceStreamer::ctrlPump()
              * @param command_idx Use integer instead of enum { "ATTACH": 0, "DETACH": 1, "SHUTDOWN": 2 }
              */
             self.ctrl_pump(command_idx);
             return py::none();
           })
      .def("get_direction_vector",
           [](InstanceStreamer& self) -> py::list {
             /**
              *
              * @brief Binding InstanceStreamer::getDirectionVector()
              */
             py::list lst;
             std::vector<double> direction_vector = self.get_direction_vector();
             for (int i = 0; i < 3; i++) {
               lst.append(direction_vector[i]);
             }
             return lst;
           })
      .def("restore",
           [](InstanceStreamer& self) -> py::none {
             /**
              *
              * @brief Binding InstanceStreamer::getDirectionVector()
              */
             self.restore();
             return py::none();
           })
      .def(
          "get_states",
          [](InstanceStreamer& self, const py::list input_thetas,
             bool verbose) -> py::object {
            // allocating memory
            double** states = (double**)malloc(6 * sizeof(double*));
            for (int i = 0; i < 6; i++) {
              states[i] = (double*)malloc(3 * sizeof(double));
            }

            if (input_thetas.size() < 6) {
              throw std::runtime_error(
                  "input_thetas_pylist must have at least 6 elements");
            }

            Kinematics::Thetas _input_thetas(
                input_thetas[0].cast<double>(), input_thetas[1].cast<double>(),
                input_thetas[2].cast<double>(), input_thetas[3].cast<double>(),
                input_thetas[4].cast<double>(), input_thetas[5].cast<double>());
            self.get_states(states, _input_thetas, verbose);

            if (states != nullptr) {
              py::list state_lst;
              for (int i = 0; i < 6; i++) {
                py::list vec;
                for (int j = 0; j < 3; j++) {
                  vec.append(states[i][j]);
                }
                state_lst.append(vec);
                delete[] states[i];
              }

              delete[] states;
              return state_lst;
            }

            {
              for (int i = 0; i < 6; i++) {
                delete[] states[i];
              }
              delete[] states;
              return py::none();
            }
          },
          py::arg("input_thetas") = py::list(), py::arg("verbose") = false)
      .def_property_readonly("get_rotplat_theta",
                             &InstanceStreamer::get_rotary_theta)
      .def("get_roboarm_thetas",
           [](InstanceStreamer& self) {
             Kinematics::Thetas roboarm_thetas = self.get_robot_arm_thetas();

             py::list lst;
             for (int i = 0; i < 6; i++) {
               lst.append(roboarm_thetas.values[i]);
             }
             return lst;
           })
      .def("get_roboarm_orientation",
           [](InstanceStreamer& self) {
             Kinematics::Orientation roboarm_orientation =
                 self.get_robot_arm_orientation();

             py::list lst;
             lst.append(roboarm_orientation.x);
             lst.append(roboarm_orientation.y);
             lst.append(roboarm_orientation.z);
             lst.append(roboarm_orientation.a);
             lst.append(roboarm_orientation.b);
             lst.append(roboarm_orientation.c);

             return lst;
           })
      .def("get_invert_axes", [](InstanceStreamer& self) {
        py::list lst;
        std::vector<int> invert_axes = self.get_invert_axes();
        for (int i = 0; i < 6; i++) {
          lst.append(invert_axes[i]);
        }
        return lst;
      });
};

// bindings.cpp ends here
