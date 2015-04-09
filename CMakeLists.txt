# start
cmake_minimum_required (VERSION 2.6)
project (KalmanTest)

# opencv
FIND_PACKAGE( OpenCV REQUIRED )

# sources
SET(KalmanTest_SOURCES main.cpp test.cpp test.h)

# include dir
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# executable and link libraries
ADD_EXECUTABLE(${PROJECT_NAME} ${KalmanTest_SOURCES})

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${OpenCV_LIBS})
