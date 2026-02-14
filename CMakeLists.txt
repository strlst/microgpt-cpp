cmake_minimum_required(VERSION 3.15...4.1)
project(
  micro_gpt
  VERSION 1.0
  LANGUAGES CXX)

add_definitions(-DCPPHTTPLIB_OPENSSL_SUPPORT)
#add_definitions(-DOPENSSL_ROOT_DIR=/usr/lib)

find_package(OpenSSL REQUIRED)
#message(STATUS "OpenSSL_LIBRARIES: ${OpenSSL_LIBRARIES}")
add_executable(micro_gpt microgpt.cpp util.cpp)
target_link_libraries(micro_gpt OpenSSL::SSL OpenSSL::Crypto)