"""Compile actual solve.cu kernel bodies against a test-only GMP backend."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build-emulation'
BUILD.mkdir(exist_ok=True)
source = (ROOT / 'solve.cu').read_text()
source = source.replace('#include <cupqc/bigint.hpp>', '#include "tests/cuda_emulation.h"')
source = source.replace('#include <cuda_runtime.h>', '')
source, count = re.subn(
    r'(\w+)<<<([^,<>]+),\s*([^<>]+)>>>\(([^;]+)\);',
    r'emulate_launch(\2,\3,[&] { \1(\4); });', source)
assert count == 6, f'Expected 6 CUDA launch sites, saw {count}; update test adapter explicitly.'
assert '<<<' not in source
(BUILD / 'solve_emulated.cpp').write_text(source)
cmd = ['clang++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-I'+str(ROOT),
       '-I/opt/homebrew/include', str(ROOT / 'main.cpp'), str(BUILD / 'solve_emulated.cpp'),
       '-L/opt/homebrew/lib', '-lgmpxx', '-lgmp', '-o', str(BUILD / 'solution')]
subprocess.run(cmd, check=True)
print(BUILD / 'solution')
