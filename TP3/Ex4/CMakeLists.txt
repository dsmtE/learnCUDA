get_filename_component(dirName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" dirName ${dirName})

set(targetName "${tpName}_${dirName}")

file(GLOB_RECURSE sources "./*.cpp" "../commonSrc/*.cpp")
file(GLOB_RECURSE headers "./*.hpp" "../commonSrc/*.hpp")
file(GLOB_RECURSE cuda "./*.cu" "../commonSrc/*.cu")

add_executable(${targetName} ${cuda} ${sources} ${headers})
target_include_directories(${targetName} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)