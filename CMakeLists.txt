cmake_minimum_required(VERSION 3.1)

project(ParkingLot_analysis VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/utils.cpp
    src/parkingdetection.cpp

    # Libs
    src/libs/tinyxml2.cpp

    # Data objects
    src/dataobjects/bbox.cpp
    
)

# Add executable
add_executable(ParkingLot_analysis ${SOURCES})

# Link libraries
target_link_libraries(ParkingLot_analysis ${OpenCV_LIBS})

# Dataset path (add as a definition)
add_definitions(-DDATA_PATH="${CMAKE_SOURCE_DIR}/dataset")

