"""Compile actual solve.cu kernel bodies against a test-only GMP backend."""
from pathlib import Path
import argparse
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build-emulation'
BUILD.mkdir(exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument('--define',action='append',default=[])
args = parser.parse_args()
source = (ROOT / 'solve.cu').read_text()
source = source.replace('#include <cupqc/bigint.hpp>', '#include "tests/cuda_emulation.h"')
source = source.replace('#include <cuda_runtime.h>', '')
# Coroutines schedule the actual kernel bodies one barrier phase at a time.
# This preserves shared-memory communication instead of erasing barriers.
matches=list(re.finditer(r'__global__ void (\w+)\([^)]*\)\s*\{',source))
for m in reversed(matches):
    start=m.end(); end=start; depth=1
    while depth:
        depth += (source[end]=='{')-(source[end]=='}')
        end += 1
    body=source[start:end-1].replace('return;', 'co_return;')
    body=body.replace('__syncthreads();','co_await std::suspend_always{};')
    decl=source[m.start():start].replace('__global__ void','EmulatedKernel')
    source=source[:m.start()]+decl+body+'co_return;\n}'+source[end:]
source, count = re.subn(
    r'(\w+)<<<([^,<>]+),\s*([^<>]+)>>>\(([^;]+)\);',
    r'emulate_launch(\2,\3,[&] { return \1(\4); });', source)
assert count == 8, f'Expected 8 CUDA launch sites, saw {count}; update test adapter explicitly.'
assert '<<<' not in source
(BUILD / 'solve_emulated.cpp').write_text(source)
cmd = ['clang++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-I'+str(ROOT),
       '-DFHERMA_PROFILE=0', *['-D'+d for d in args.define],
       '-I/opt/homebrew/include', str(ROOT / 'main.cpp'), str(BUILD / 'solve_emulated.cpp'),
       '-L/opt/homebrew/lib', '-lgmpxx', '-lgmp', '-o', str(BUILD / 'solution')]
subprocess.run(cmd, check=True)
print(BUILD / 'solution')
