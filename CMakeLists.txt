cmake_minimum_required(VERSION 3.9.5)
 
project(ReadWrite)
 
FIND_PACKAGE(ITK)

IF(ITK_FOUND)
    
  INCLUDE(${ITK_USE_FILE})

ELSE(ITK_FOUND)
    
  MESSAGE(FATAL_ERROR
    "ITK not found. Please set ITK_DIR.")

ENDIF(ITK_FOUND)
 
add_executable(ReadWrite ReadWrite.cxx getopt.c getopt.h)
target_link_libraries(ReadWrite ${ITK_LIBRARIES})