set(TARGET tensorrelay_stage_executor)

add_library(${TARGET} SHARED tensorrelay-stage-executor.cpp)
if(WIN32)
    set_target_properties(${TARGET} PROPERTIES PREFIX "")
endif()
target_include_directories(${TARGET} PRIVATE ../../include ../../vendor)
# llama-common provides the tools-aware chat machinery (jinja templating, PEG
# tool-call parsing) behind the ABI v10 chat_begin/chat_parse entries.
target_link_libraries(${TARGET} PRIVATE llama llama-common)
target_compile_definitions(${TARGET} PRIVATE TENSORRELAY_STAGE_EXECUTOR_BUILD)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY RUNTIME ARCHIVE)
endif()

# Diagnostic-only. This target does not link the stage executor and does not
# decide which ops are allowed.
add_executable(test-tensorrelay-policy-diag
    test-tensorrelay-policy-diag.cpp
    ../../ggml/src/ggml-tensorrelay-policy-diag.cpp)
target_include_directories(test-tensorrelay-policy-diag PRIVATE ../../ggml/include)
target_compile_features(test-tensorrelay-policy-diag PRIVATE cxx_std_17)
find_package(Threads REQUIRED)
target_link_libraries(test-tensorrelay-policy-diag PRIVATE Threads::Threads)
add_test(NAME test-tensorrelay-policy-diag COMMAND test-tensorrelay-policy-diag)
