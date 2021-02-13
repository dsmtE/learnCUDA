get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" dirName ${dirName})

set(targetName "${tpName}_${dirName}")

file(GLOB_RECURSE sources "src/*.cpp" "../commonSrc/*.cpp")
file(GLOB_RECURSE headers "src/*.hpp" "../commonSrc/*.hpp")
file(GLOB_RECURSE cuda "src/*.cu" "../commonSrc/*.cu")

message("target : ${targetName}")
add_executable(${targetName} ${cuda} ${sources} ${headers})
target_include_directories(${targetName} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)