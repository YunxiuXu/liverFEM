# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-src")
  file(MAKE_DIRECTORY "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-src")
endif()
file(MAKE_DIRECTORY
  "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-build"
  "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix"
  "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix/tmp"
  "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix/src/predicates-populate-stamp"
  "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix/src"
  "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix/src/predicates-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix/src/predicates-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/yunxiuxu/Documents/tetfemcpp/VegaFEM/build/macos-release/_deps/predicates-subbuild/predicates-populate-prefix/src/predicates-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
