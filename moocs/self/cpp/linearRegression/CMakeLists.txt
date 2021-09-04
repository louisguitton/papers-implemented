cmake_minimum_required(VERSION 3.0.0)
project(linearRegression VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(linearRegression main.cpp)
target_sources(linearRegression
    PRIVATE
        "main.cpp"
        "utilities.cpp"
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
