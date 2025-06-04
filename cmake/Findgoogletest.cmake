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

find_library(GTEST_LIBRARY
  NAMES gtest
  HINTS
    ENV SPACK_ENV
    ${CMAKE_PREFIX_PATH}
    /opt/googletest
  PATH_SUFFIXES lib lib64
)

# Create IMPORTED target
if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY)
  add_library(GTest::gtest SHARED IMPORTED)
  set_target_properties(GTest::gtest PROPERTIES
    IMPORTED_LOCATION "${GTEST_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
  )
  set(googletest_FOUND TRUE)
else()
  set(googletest_FOUND FALSE)
endif()
