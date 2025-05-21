cmake_minimum_required(VERSION 3.16)

# Find HDF5 package
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Define the project name
project(MicroMOC VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the executable
add_executable(serial_moc.exe serial_moc.cpp)

# Optional: Specify output directory
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Optional: Add include directories if needed
# target_include_directories(serial_moc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link with HDF5 libraries
message(STATUS "HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
message(STATUS "HDF5_DEFINITIONS: ${HDF5_DEFINITIONS}")
target_link_libraries(serial_moc.exe PRIVATE ${HDF5_LIBRARIES})
target_include_directories(serial_moc.exe PRIVATE ${HDF5_INCLUDE_DIRS})
target_compile_definitions(serial_moc.exe PRIVATE ${HDF5_DEFINITIONS})