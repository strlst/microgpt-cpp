cmake_minimum_required(VERSION 3.15...4.1)
project(
  micro_gpt
  VERSION 1.0
  LANGUAGES CXX)

add_definitions(-DCPPHTTPLIB_OPENSSL_SUPPORT)
#add_definitions(-DOPENSSL_ROOT_DIR=/usr/lib)
set_source_files_properties(lib/httplib.h PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")

find_package(OpenSSL REQUIRED)
#message(STATUS "OpenSSL_LIBRARIES: ${OpenSSL_LIBRARIES}")
add_executable(micro_gpt src/microgpt.cpp src/util.cpp src/value.cpp src/model.cpp src/adam.cpp)
target_link_libraries(micro_gpt OpenSSL::SSL OpenSSL::Crypto)