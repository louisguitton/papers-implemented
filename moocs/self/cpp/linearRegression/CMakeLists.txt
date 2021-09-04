cmake_minimum_required(VERSION 3.0.0)

project(linearRegression VERSION 0.1.0)

find_package(unofficial-sqlite3 CONFIG REQUIRED)

add_executable(linearRegression src/main.cpp)

target_link_libraries(linearRegression
    PRIVATE
        unofficial::sqlite3::sqlite3)
