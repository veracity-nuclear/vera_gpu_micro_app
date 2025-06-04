# Fallback CMake module to locate googletest and export an interface target

# Try to find the main googletest header
find_path(GTEST_INCLUDE_DIR
  NAMES
    gtest/gtest.h
  HINTS
    ENV SPACK_ENV
    ${CMAKE_PREFIX_PATH}
    /opt/googletest
  PATH_SUFFIXES include include/gtest
)

# Create an INTERFACE imported target for googletest
#add_library(GTest::gtest INTERFACE IMPORTED)

# Specify include directories
#target_include_directories(GTest::gtest INTERFACE
#  ${GTEST_INCLUDE_DIR}
#)

find_library(GTEST_LIBRARY
  NAMES gtest
  HINTS
    ENV SPACK_ENV
    ${CMAKE_PREFIX_PATH}
    /opt/googletest
  PATH_SUFFIXES lib lib64
)

message(STATUS "GTest include dir: ${GTEST_INCLUDE_DIR}")
message(STATUS "GTest library: ${GTEST_LIBRARY}")

# Create IMPORTED target
if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY)
  add_library(GTest::gtest SHARED IMPORTED)
  set_target_properties(GTest::gtest PROPERTIES
    IMPORTED_LOCATION "${GTEST_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
  )
else()
  message(FATAL_ERROR "Failed to find GTest include and/or library.")
endif()
