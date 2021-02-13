get_filename_component(tpName ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" tpName ${tpName})

SUBDIRLIST(subDirs ${CMAKE_CURRENT_SOURCE_DIR})

FOREACH(subdir ${subDirs})
  ADD_SUBDIRECTORY(${subdir})
ENDFOREACH()
