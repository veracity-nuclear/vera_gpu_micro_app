cmake_minimum_required(VERSION 3.16)

# Find HDF5 package
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Option 1: Add the path to the CMake modules directory where Findhighfive.cmake is located
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(highfive REQUIRED)

# Define the project name
project(MicroMOC VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
# Find all cpp files in the current directory
file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Add the executable using the globbed sources
add_executable(serial_moc.exe ${CPP_SOURCES})

# Optional: Specify output directory
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Optional: Add include directories if needed
# target_include_directories(serial_moc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link with HDF5 libraries
target_link_libraries(serial_moc.exe PRIVATE ${HDF5_LIBRARIES} ${highfive_LIBRARIES})
target_include_directories(serial_moc.exe PRIVATE ${HDF5_INCLUDE_DIRS} ${highfive_INCLUDE_DIRS})
target_compile_definitions(serial_moc.exe PRIVATE ${HDF5_DEFINITIONS} ${highfive_DEFINITIONS})