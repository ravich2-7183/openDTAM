cmake_minimum_required (VERSION 2.6)
project(Test_Dense_Mapper)

Set(CMAKE_BUILD_TYPE Debug) 
cmake_policy(SET CMP0012 NEW)

# ROS setup
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
# rosbuild_init()
# set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)
# set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb -std=c++11")

find_package(OpenCV 2.4.9 EXACT REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system)

###############################################################################
macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND DTAM_SOURCES "${_relPath}/${_src}")
        else()
            list (APPEND DTAM_SOURCES "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (DTAM_SOURCES ${DTAM_SOURCES} PARENT_SCOPE)
    endif()
endmacro(add_sources)
###############################################################################

set(BASEPATH "${CMAKE_SOURCE_DIR}/src")

include_directories("${BASEPATH}" ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS}) # ${PCL_INCLUDE_DIRS})

add_sources(../CostVolume.cpp
			../CostVolume_minimizeA.cu
			../CostVolume_update.cu
            ./src/test_texture.cu
			../../Regulariser/Regulariser.cpp
			../../Regulariser/Regulariser_computeG.cu
			../../Regulariser/Regulariser_update_q_d.cu)

include_directories("${BASEPATH}" ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} )

# Future proofing gpu code by forcing device independent PTX code to be generated, instead of gpu specific cubin code. See:
# https://devtalk.nvidia.com/default/topic/938165/when-will-cuda-8-be-released-1080-can-t-run-with-7-5/
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_30,code=compute_30; -gencode arch=compute_35,code=compute_35;--use_fast_math;--restrict; -lineinfo; -g; -G; -O3)
cuda_add_library(openDTAM_LIB SHARED ${DTAM_SOURCES})
target_link_libraries(openDTAM_LIB ${OpenCV_LIBS} ${Boost_LIBRARIES})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
add_executable(test-dense-mapping src/test-depth-estimation.cpp)
target_link_libraries(test-dense-mapping ${OpenCV_LIBS} ${Boost_LIBRARIES} openDTAM_LIB)

add_executable(test-multiple-images src/test-multiple-images.cpp)
target_link_libraries(test-multiple-images ${OpenCV_LIBS} ${Boost_LIBRARIES} openDTAM_LIB)

# add_executable(test-multiple-images-ahanda src/test-multiple-images-ahanda.cpp)
# target_link_libraries(test-multiple-images-ahanda ${OpenCV_LIBS} ${Boost_LIBRARIES} openDTAM_LIB)

