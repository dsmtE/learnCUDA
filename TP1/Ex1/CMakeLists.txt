get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" dirName ${dirName})

set(targetName "${tpName}_${dirName}")

file(GLOB_RECURSE sources "src/*.cpp")
file(GLOB_RECURSE headers "src/*.hpp")
file(GLOB_RECURSE cuda "src/*.cu")

message("target : ${targetName}")
add_executable(${targetName} ${cuda} ${sources} ${headers})