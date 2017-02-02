cmake_minimum_required(VERSION 3.6)
project(artmap_c)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
    artmap.cpp
    input.dat
    Makefile
    output.dat
    README
    te_input.dat
    te_output.dat)

add_executable(artmap_c ${SOURCE_FILES})