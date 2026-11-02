# Vulkan builds with bounded memory

For small CI workers, configure `-DGGML_VULKAN_SHADER_JOBS=1 -DGGML_VULKAN_SHADER_PARTS=16` and build with `cmake --build ... -j1`.

SHADER_JOBS limits the generator's internal glslc subprocess pool, which is separate from the outer CMake job count. Zero preserves automatic selection (up to 16). SHADER_PARTS distributes complete embedded SPIR-V arrays between a fixed number of C++ translation units, balancing their byte counts. All bytes, symbols, lengths and supported shaders are retained. The default of one preserves the existing file layout. The extra files are declared CMake outputs and target sources, so incremental regeneration and both Make/Ninja builds track them.

This trades additional compilation steps for lower peak compiler memory. It does not impose a new runtime GPU limit or disable Vulkan features. Keep the outer build serial when using a small memory budget. Run `python3 tests/test-vulkan-memory.py` for a compile/link payload and concurrency regression test.
