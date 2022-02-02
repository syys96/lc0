project(Botzone)
cmake_minimum_required(VERSION 3.15.3)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

add_executable(bot bot.cpp)
add_executable(allInOne bot_merged_v2.cpp)