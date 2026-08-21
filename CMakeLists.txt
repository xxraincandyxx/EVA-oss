cmake_minimum_required(VERSION 3.20)

project(EVA VERSION 2.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EVA_BUILD_PYTHON "Build the Python bindings" ON)

set(EVA_CONTROL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dynamo_")
file(GLOB_RECURSE EVA_CONTROL_SOURCES CONFIGURE_DEPENDS
  "${EVA_CONTROL_ROOT}/src/*.cpp"
)

add_library(eva_control SHARED ${EVA_CONTROL_SOURCES})
target_include_directories(eva_control PUBLIC "${EVA_CONTROL_ROOT}/include")
target_compile_features(eva_control PUBLIC cxx_std_17)

if(EVA_BUILD_PYTHON)
  find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_DISCOVERY_STATUS
  )
  if(PYBIND11_DISCOVERY_STATUS EQUAL 0 AND PYBIND11_CMAKE_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(dynamo_ src/backend/bindings/bindings.cpp)
  target_link_libraries(dynamo_ PRIVATE eva_control)
  install(TARGETS dynamo_ LIBRARY DESTINATION backend/lib)
endif()

if(BUILD_TESTING)
  add_executable(eva_kinematics_test
    "${EVA_CONTROL_ROOT}/test/test_kinematics.cpp"
  )
  target_link_libraries(eva_kinematics_test PRIVATE eva_control)
  add_test(NAME kinematics COMMAND eva_kinematics_test)

  add_executable(eva_integration_test
    "${EVA_CONTROL_ROOT}/test/test_main.cpp"
  )
  target_link_libraries(eva_integration_test PRIVATE eva_control)
  add_test(NAME integration COMMAND eva_integration_test)
endif()
