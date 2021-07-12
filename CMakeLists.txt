cmake_minimum_required(VERSION 3.11)

project(racon VERSION 3.0.3
              LANGUAGES CXX
              DESCRIPTION "Racon is a c++ consensus module for raw de novo DNA assembly of long uncorrected reads.")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

include(FetchContent)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(racon_main_project ON)
endif ()
option(racon_install "Generate library install target" OFF)
option(racon_build_exe "Build executable" ${racon_main_project})
option(racon_build_tests "Build unit tests" ${racon_main_project})
option(racon_enable_cuda "Build with NVIDIA CUDA support" OFF)

find_package(edlib 1.2.6 QUIET)
if (NOT edlib_FOUND)
  if (racon_install)
    message(FATAL_ERROR "Missing package edlib!")
  endif ()

  FetchContent_Declare(
    edlib
    GIT_REPOSITORY https://github.com/martinsos/edlib
    GIT_TAG d5774b4)

  FetchContent_GetProperties(edlib)
  if (NOT edlib_POPULATED)
    FetchContent_Populate(edlib)
    add_subdirectory(
      ${edlib_SOURCE_DIR}
      ${edlib_BINARY_DIR}
      EXCLUDE_FROM_ALL)
  endif ()
endif ()

find_package(ram 2.1.1 QUIET)
if (NOT ram_FOUND)
  if (racon_install)
    message(FATAL_ERROR "Missing package ram!")
  endif ()

  FetchContent_Declare(
    ram
    GIT_REPOSITORY https://github.com/lbcb-sci/ram
    GIT_TAG 2.1.1)

  FetchContent_GetProperties(ram)
  if (NOT ram_POPULATED)
    FetchContent_Populate(ram)
    add_subdirectory(
      ${ram_SOURCE_DIR}
      ${ram_BINARY_DIR}
      EXCLUDE_FROM_ALL)
  endif ()
endif ()

find_package(spoa 4.0.8 QUIET)
if (NOT spoa_FOUND)
  if (racon_install)
    message(FATAL_ERROR "Missing package spoa!")
  endif ()

  FetchContent_Declare(
    spoa
    GIT_REPOSITORY https://github.com/rvaser/spoa
    GIT_TAG 464ec7d)

  FetchContent_GetProperties(spoa)
  if (NOT spoa_POPULATED)
    FetchContent_Populate(spoa)
    add_subdirectory(
      ${spoa_SOURCE_DIR}
      ${spoa_BINARY_DIR}
      EXCLUDE_FROM_ALL)
  endif ()
endif ()

if (racon_build_exe OR racon_build_tests)
  find_package(bioparser 3.0.13 QUIET)
  if (NOT bioparser_FOUND)
    FetchContent_Declare(
      bioparser
      GIT_REPOSITORY https://github.com/rvaser/bioparser
      GIT_TAG 3.0.13)

    FetchContent_GetProperties(bioparser)
    if (NOT bioparser_POPULATED)
      FetchContent_Populate(bioparser)
      add_subdirectory(
        ${bioparser_SOURCE_DIR}
        ${bioparser_BINARY_DIR}
        EXCLUDE_FROM_ALL)
    endif ()
  endif ()
endif ()

if (racon_build_tests)
  find_package(GTest 1.10.0 QUIET)
  if (NOT GTest_FOUND)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest
      GIT_TAG release-1.10.0)

    FetchContent_GetProperties(googletest)
    if (NOT googletest_POPULATED)
      FetchContent_Populate(googletest)
      add_subdirectory(
        ${googletest_SOURCE_DIR}
        ${googletest_BINARY_DIR}
        EXCLUDE_FROM_ALL)
      add_library(GTest::Main ALIAS gtest_main)
    endif ()
  endif ()
endif ()

if (racon_enable_cuda)
  find_package(CUDA 9.0 QUIET REQUIRED)
  if (NOT ${CUDA_FOUND})
    message(FATAL_ERROR "CUDA not detected on system. Please install")
  else ()
    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
  endif ()
  if (DEFINED CLARAGENOMICSANALYSIS_SDK_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${CLARAGENOMICSANALYSIS_SDK_PATH}/cmake")
    find_package(cudapoa REQUIRED)
    find_package(cudaaligner REQUIRED)
  elseif (DEFINED CLARAGENOMICSANALYSIS_SRC_PATH)
    if (NOT TARGET cudapoa)
      add_subdirectory(
        ${CLARAGENOMICSANALYSIS_SRC_PATH}
        ${CMAKE_CURRENT_BINARY_DIR}/GenomeWorks
        EXCLUDE_FROM_ALL)
    endif ()
    if (NOT TARGET cudaaligner)
      add_subdirectory(
        ${CLARAGENOMICSANALYSIS_SRC_PATH}
        ${CMAKE_CURRENT_BINARY_DIR}/GenomeWorks
        EXCLUDE_FROM_ALL)
    endif ()
  else ()
    if (racon_install)
      message(FATAL_ERROR "Missing package GenomeWorks!")
    endif ()
    FetchContent_Declare(
      genomeworks
      GIT_REPOSITORY https://github.com/clara-parabricks/GenomeWorks
      GIT_TAG v0.5.3)

    FetchContent_GetProperties(genomeworks)
    if (NOT genomeworks_POPULATED)
      FetchContent_Populate(genomeworks)
      add_subdirectory(
        ${genomeworks_SOURCE_DIR}
        ${genomeworks_BINARY_DIR}
        EXCLUDE_FROM_ALL)
    endif ()
  endif ()
endif ()

set(racon_sources
  src/polisher.cpp
  src/overlap.cpp
  src/window.cpp)

if(racon_enable_cuda)
  include_directories(${PROJECT_SOURCE_DIR}/src)
  list(APPEND racon_sources
    src/cuda/cudaaligner.cpp
    src/cuda/cudabatch.cpp
    src/cuda/cudapolisher.cpp)

  cuda_add_library(racon
    ${racon_sources})

  target_compile_definitions(racon
    PRIVATE CUDA_ENABLED)
else ()
  add_library(racon
    ${racon_sources})
endif ()
add_library(racon::racon ALIAS racon)

target_link_libraries(racon
  edlib::edlib
  spoa::spoa
  ram::ram)

if (racon_enable_cuda)
  target_link_libraries(racon
    cudapoa
    cudaaligner)
endif ()

target_include_directories(racon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

if (racon_install)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

  install(
    TARGETS racon
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(
    DIRECTORY include/racon
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(
    EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif ()

if (racon_build_exe)
  add_executable(racon_exe
    src/main.cpp)

  target_link_libraries(racon_exe
    racon
    bioparser::bioparser)

  target_compile_definitions(racon_exe PRIVATE VERSION="${PROJECT_VERSION}")
  set_property(TARGET racon_exe PROPERTY OUTPUT_NAME racon)

  if (racon_enable_cuda)
    target_compile_definitions(racon_exe PRIVATE CUDA_ENABLED)
  endif ()

  install(TARGETS racon_exe DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()

if (racon_build_tests)
  add_executable(racon_test
    test/polisher_test.cpp)

  target_link_libraries(racon_test
    racon
    bioparser::bioparser
    GTest::Main)

  target_compile_definitions(racon_test
    PRIVATE TEST_DATA="${PROJECT_SOURCE_DIR}/test/data/")

  if (racon_enable_cuda)
    target_compile_definitions(racon_test PRIVATE CUDA_ENABLED)
  endif ()
endif ()
