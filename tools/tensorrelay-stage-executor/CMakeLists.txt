set(TARGET tensorrelay_stage_executor)

add_library(${TARGET} SHARED tensorrelay-stage-executor.cpp)
if(WIN32)
    set_target_properties(${TARGET} PROPERTIES PREFIX "")
endif()
target_include_directories(${TARGET} PRIVATE ../../include ../../vendor)
target_link_libraries(${TARGET} PRIVATE llama)
target_compile_definitions(${TARGET} PRIVATE TENSORRELAY_STAGE_EXECUTOR_BUILD)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY RUNTIME ARCHIVE)
endif()
