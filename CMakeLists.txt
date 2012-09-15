project(ICuCme)
cmake_minimum_required(VERSION 2.6)


set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_BUILD_TYPE Release)


include_directories(
  #/usr/local/include 
  /opt/ros/fuerte/include
)

link_directories(
  /opt/ros/fuerte/lib
    /usr/lib
    #/usr/local/lib 
)


add_executable(ICuCme src/ICuCme.cpp)
target_link_libraries(ICuCme opencv_core opencv_highgui opencv_imgproc opencv_objdetect)
