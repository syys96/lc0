project(Botzone)
cmake_minimum_required(VERSION 3.15.3)

add_executable(bot bot.cpp)
add_executable(allInOne bot_merged.cpp)