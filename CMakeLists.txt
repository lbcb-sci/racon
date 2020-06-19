cmake_minimum_required(VERSION 3.9)

project(racon VERSION 2.0.8
              LANGUAGES CXX
              DESCRIPTION "Racon is a c++ consensus module for raw de novo DNA assembly of long uncorrected reads.")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

if (NOT TARGET edlib)
  add_subdirectory(vendor/edlib EXCLUDE_FROM_ALL)
endif ()
if (NOT TARGET spoa)
  add_subdirectory(vendor/spoa EXCLUDE_FROM_ALL)
endif ()
if (NOT TARGET ram)
  add_subdirectory(vendor/ram EXCLUDE_FROM_ALL)
endif ()
if (NOT TARGET bioparser)
  add_subdirectory(vendor/ram/vendor/bioparser EXCLUDE_FROM_ALL)
endif ()

option(racon_enable_cuda "Build racon with NVIDIA CUDA support" OFF)
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
      add_subdirectory(${CLARAGENOMICSANALYSIS_SRC_PATH} ${CMAKE_CURRENT_BINARY_DIR}/ClaraGenomicsAnalysis EXCLUDE_FROM_ALL)
    endif ()
    if (NOT TARGET cudaaligner)
      add_subdirectory(${CLARAGENOMICSANALYSIS_SRC_PATH} ${CMAKE_CURRENT_BINARY_DIR}/ClaraGenomicsAnalysis EXCLUDE_FROM_ALL)
    endif ()
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ClaraGenomicsAnalysis)
    if (NOT TARGET cudapoa)
      add_subdirectory(vendor/ClaraGenomicsAnalysis ${CMAKE_CURRENT_BINARY_DIR}/ClaraGenomicsAnalysis EXCLUDE_FROM_ALL)
    endif ()
    if (NOT TARGET cudaaligner)
      add_subdirectory(vendor/ClaraGenomicsAnalysis ${CMAKE_CURRENT_BINARY_DIR}/ClaraGenomicsAnalysis EXCLUDE_FROM_ALL)
    endif ()
  else ()
    if (NOT TARGET cudapoa)
      add_subdirectory(../ClaraGenomicsAnalysis ${CMAKE_CURRENT_BINARY_DIR}/ClaraGenomicsAnalysis EXCLUDE_FROM_ALL)
    endif ()
    if (NOT TARGET cudaaligner)
      add_subdirectory(../ClaraGenomicsAnalysis ${CMAKE_CURRENT_BINARY_DIR}/ClaraGenomicsAnalysis EXCLUDE_FROM_ALL)
    endif ()
  endif ()
endif ()

include_directories(${PROJECT_SOURCE_DIR}/src)
set(${PROJECT_NAME}_sources
  src/polisher.cpp
  src/overlap.cpp
  src/window.cpp)

if(racon_enable_cuda)
  list(APPEND ${PROJECT_NAME}_sources
    src/cuda/cudaaligner.cpp
    src/cuda/cudabatch.cpp
    src/cuda/cudapolisher.cpp)
  cuda_add_library(${PROJECT_NAME} ${${PROJECT_NAME}_sources})
  target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ENABLED)
else()
  add_library(${PROJECT_NAME} ${${PROJECT_NAME}_sources})
endif()
target_link_libraries(${PROJECT_NAME} edlib_static spoa ram)
if (racon_enable_cuda)
  target_link_libraries(${PROJECT_NAME} cudapoa cudaaligner)
endif ()

target_include_directories(${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

option(racon_build_executable "Build racon executable" OFF)
if (racon_build_executable)
  if (NOT TARGET bioparser)
    add_subdirectory(vendor/ram/vendor/bioparser EXCLUDE_FROM_ALL)
  endif ()
  add_executable(${PROJECT_NAME}_exe src/main.cpp)
  target_link_libraries(${PROJECT_NAME}_exe ${PROJECT_NAME} bioparser)
  target_compile_definitions(${PROJECT_NAME}_exe
    PRIVATE RACON_VERSION="v${PROJECT_VERSION}")
  set_target_properties(${PROJECT_NAME}_exe
    PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
  if (racon_enable_cuda)
    target_compile_definitions(${PROJECT_NAME}_exe PRIVATE CUDA_ENABLED)
  endif()

  include(GNUInstallDirs)
  install(TARGETS ${PROJECT_NAME}_exe DESTINATION ${CMAKE_INSTALL_BINDIR})
endif ()

option(racon_build_tests "Build racon unit tests" OFF)
if (racon_build_tests)
  find_package(GTest REQUIRED)
  add_executable(${PROJECT_NAME}_test test/polisher_test.cpp)
  target_link_libraries(${PROJECT_NAME}_test ${PROJECT_NAME} bioparser GTest::Main)
  target_compile_definitions(${PROJECT_NAME}_test
    PRIVATE RACON_DATA_PATH="${PROJECT_SOURCE_DIR}/test/data/")
  if (racon_enable_cuda)
    target_compile_definitions(${PROJECT_NAME}_test PRIVATE CUDA_ENABLED)
  endif()
endif ()
