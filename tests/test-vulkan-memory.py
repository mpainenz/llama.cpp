#!/usr/bin/env python3
"""Compile/link split shader data and exercise the real concurrency limiter."""
import pathlib, subprocess, tempfile
root=pathlib.Path(__file__).resolve().parents[1]
source=root/'ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp'
with tempfile.TemporaryDirectory() as tmp:
    p=pathlib.Path(tmp)
    harness='''#define main generator_main
#include "SOURCE"
#undef main
int main(int argc, char **argv) {
 shader_jobs=1;
 std::atomic<int> active{0}, peak{0};
 std::vector<std::thread> threads;
 for(int i=0;i<8;i++) threads.emplace_back([&]{ auto slot=acquire_compile_slot(); int n=++active; peak.store(std::max(peak.load(),n)); std::this_thread::sleep_for(std::chrono::milliseconds(5)); --active; });
 for(auto & t:threads) t.join();
 if(peak!=1) return 2;
 output_dir=argv[1]; target_hpp=output_dir+"/shaders.hpp";
 for(int n=0;n<19;n++){std::string name="unit_"+std::to_string(n); std::string path=output_dir+"/"+name+".spv"; write_binary_file(path,std::string(n+1,char(n)));shader_fnames.emplace_back(name,path);}
 write_output_files();
 input_filepath="unit.comp"; target_cpp=output_dir+"/shaders.cpp";shader_parts=16;write_output_files();
 return 0;
}
'''.replace('SOURCE',str(source))
    (p/'h.cpp').write_text(harness)
    subprocess.run(['c++','-std=c++17','-pthread','-O0',str(p/'h.cpp'),'-o',str(p/'h')],check=True)
    subprocess.run([str(p/'h'),tmp],check=True)
    checks='\n'.join(f'if(unit_{n}_len != {n+1}) return 1; for(unsigned i=0;i<unit_{n}_len;i++) if(unit_{n}_data[i]!={n}) return 2;' for n in range(19))
    (p/'verify.cpp').write_text('#include "shaders.hpp"\nint main(){'+checks+'}\n')
    files=[p/'shaders.cpp']+sorted(p.glob('shaders.cpp.part*.cpp'))
    assert len(files)==16
    subprocess.run(['c++','-std=c++17',*(str(f) for f in files),str(p/'verify.cpp'),'-o',str(p/'verify')],check=True)
    subprocess.run([str(p/'verify')],check=True)
print('PASS: one shader worker; all split payload symbols and bytes preserved across 16 linked units')
