cmake_minimum_required(VERSION 3.15...4.1)
project(
  microgpt
  VERSION 1.0
  LANGUAGES CXX)

add_definitions(-DCPPHTTPLIB_OPENSSL_SUPPORT)
#add_definitions(-DOPENSSL_ROOT_DIR=/usr/lib)
set_source_files_properties(lib/httplib.h PROPERTIES COMPILE_FLAGS "-Wno-deprecated-declarations")

find_package(OpenSSL REQUIRED)
#message(STATUS "OpenSSL_LIBRARIES: ${OpenSSL_LIBRARIES}")

#find_package(OpenMP REQUIRED)

# uncomment to enable gprof profiling
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")

add_executable(microgpt src/microgpt.cpp src/util.cpp src/value.cpp src/model.cpp src/adam.cpp)
target_link_libraries(microgpt OpenSSL::SSL OpenSSL::Crypto)
#if(OpenMP_CXX_FOUND)
    #target_link_libraries(microgpt OpenMP::OpenMP_CXX)
    #target_compile_options(microgpt PRIVATE ${OpenMP_CXX_FLAGS})
#endif()