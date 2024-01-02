project(my-project)

cmake_minimum_required(VERSION 3.16)

set(TFHE_C_API "./tfhe-rs/target/release")

include_directories(${TFHE_C_API})
add_library(tfhe STATIC IMPORTED)
set_target_properties(tfhe PROPERTIES IMPORTED_LOCATION ${TFHE_C_API}/libtfhe.a)

if(APPLE)
    find_library(SECURITY_FRAMEWORK Security)
    if (NOT SECURITY_FRAMEWORK)
        message(FATAL_ERROR "Security framework not found")
    endif()
endif()

set(EXECUTABLE_NAME my-executable)
add_executable(${EXECUTABLE_NAME} main.c)
target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${EXECUTABLE_NAME} LINK_PUBLIC tfhe m pthread dl)
if(APPLE)
    target_link_libraries(${EXECUTABLE_NAME} LINK_PUBLIC ${SECURITY_FRAMEWORK})
endif()
target_compile_options(${EXECUTABLE_NAME} PRIVATE -Werror)
