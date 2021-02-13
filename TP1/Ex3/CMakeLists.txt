get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" dirName ${dirName})

set(targetName "${tpName}_${dirName}")

file(GLOB_RECURSE sources "src/*")

add_executable(${targetName} ${sources})
target_include_directories(${targetName} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")