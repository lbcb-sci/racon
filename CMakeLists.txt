cmake_minimum_required(VERSION 3.11)

project(racon VERSION 1.5.0
              LANGUAGES CXX
              DESCRIPTION "Racon is a consensus module for de novo genome assembly.")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

include(FetchContent)
include(GNUInstallDirs)

if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(racon_main_project ON)
endif ()
option(racon_build_tests "Build unit tests" ${racon_main_project})
option(racon_build_wrapper "Build wrapper" OFF)
option(racon_enable_cuda "Build with NVIDIA CUDA support" OFF)

find_package(bioparser 3.0.15 QUIET)
if (NOT bioparser_FOUND)
  FetchContent_Declare(
    bioparser
    GIT_REPOSITORY https://github.com/rvaser/bioparser
    GIT_TAG 3.0.15)

  FetchContent_GetProperties(bioparser)
  if (NOT bioparser_POPULATED)
    FetchContent_Populate(bioparser)
    add_subdirectory(
      ${bioparser_SOURCE_DIR}
      ${bioparser_BINARY_DIR}
      EXCLUDE_FROM_ALL)
  endif ()
endif ()

find_package(edlib 1.2.7 QUIET)
if (NOT edlib_FOUND)
  FetchContent_Declare(
    edlib
    GIT_REPOSITORY https://github.com/martinsos/edlib
    GIT_TAG v1.2.7)

  FetchContent_GetProperties(edlib)
  if (NOT edlib_POPULATED)
    FetchContent_Populate(edlib)
    add_subdirectory(
      ${edlib_SOURCE_DIR}
      ${edlib_BINARY_DIR}
      EXCLUDE_FROM_ALL)
  endif ()
endif ()

find_package(spoa 4.0.8 QUIET)
if (NOT spoa_FOUND)
  FetchContent_Declare(
    spoa
    GIT_REPOSITORY https://github.com/rvaser/spoa
    GIT_TAG 4.0.8)

  FetchContent_GetProperties(spoa)
  if (NOT spoa_POPULATED)
    FetchContent_Populate(spoa)
    add_subdirectory(
      ${spoa_SOURCE_DIR}
      ${spoa_BINARY_DIR}
      EXCLUDE_FROM_ALL)
  endif ()
endif ()

find_package(thread_pool 4.0.0 QUIET)
if (NOT thread_pool_FOUND)
  FetchContent_Declare(
    thread_pool
    GIT_REPOSITORY https://github.com/rvaser/thread_pool
    GIT_TAG 4.0.0)

  FetchContent_GetProperties(thread_pool)
  if (NOT thread_pool_POPULATED)
    FetchContent_Populate(thread_pool)
    add_subdirectory(
      ${thread_pool_SOURCE_DIR}
      ${thread_pool_BINARY_DIR}
      EXCLUDE_FROM_ALL)
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

if (racon_build_wrapper)
  find_package(rampler 2.0.0 QUIET)
  if (NOT rampler_FOUND)
    FetchContent_Declare(
      rampler
      GIT_REPOSITORY https://github.com/rvaser/rampler
      GIT_TAG 2.0.0)

    FetchContent_GetProperties(rampler)
    if (NOT rampler_POPULATED)
      FetchContent_Populate(rampler)
      add_subdirectory(
        ${rampler_SOURCE_DIR}
        ${rampler_BINARY_DIR})
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
  src/logger.cpp
  src/polisher.cpp
  src/overlap.cpp
  src/sequence.cpp
  src/window.cpp)

if (racon_enable_cuda)
  include_directories(${PROJECT_SOURCE_DIR}/src)
  list(APPEND racon_sources
    src/cuda/cudapolisher.cpp
    src/cuda/cudabatch.cpp
    src/cuda/cudaaligner.cpp)

  cuda_add_library(racon
    ${racon_sources})

  target_compile_definitions(racon
    PRIVATE CUDA_ENABLED)
else ()
  add_library(racon
    ${racon_sources})
endif ()

target_link_libraries(racon
  bioparser::bioparser
  edlib::edlib
  spoa::spoa
  thread_pool::thread_pool)

if (racon_enable_cuda)
  target_link_libraries(racon
    cudapoa
    cudaaligner)
endif ()

target_include_directories(racon PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)

add_executable(racon_exe
  src/main.cpp)

target_link_libraries(racon_exe
  racon)

if (racon_enable_cuda)
  target_compile_definitions(racon_exe
    PRIVATE CUDA_ENABLED)
endif ()

target_compile_definitions(racon_exe PRIVATE VERSION="${PROJECT_VERSION}")
set_property(TARGET racon_exe PROPERTY OUTPUT_NAME racon)

install(TARGETS racon_exe DESTINATION ${CMAKE_INSTALL_BINDIR})

if (racon_build_tests)
  add_executable(racon_test
    test/racon_test.cpp)

  target_link_libraries(racon_test
    racon
    GTest::Main)

  target_compile_definitions(racon_test
    PRIVATE TEST_DATA="${PROJECT_SOURCE_DIR}/test/data/")

  if (racon_enable_cuda)
    target_compile_definitions(racon_test
      PRIVATE CUDA_ENABLED)
  endif ()
endif()

if (racon_build_wrapper)
  set(racon_path ${PROJECT_BINARY_DIR}/bin/racon)
  set(rampler_path ${PROJECT_BINARY_DIR}/_deps/rampler-build/bin/rampler)
  if (racon_enable_cuda)
    set(racon_wrapper_enable_cuda True)
  else ()
    set(racon_wrapper_enable_cuda False)
  endif ()
  configure_file(${PROJECT_SOURCE_DIR}/scripts/racon_wrapper.py
    ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/racon_wrapper)
  file(COPY ${PROJECT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/racon_wrapper
    DESTINATION ${PROJECT_BINARY_DIR}/bin
    FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)
endif()

# Add Debian packaging
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Robert Vaser")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
include(CPack)
