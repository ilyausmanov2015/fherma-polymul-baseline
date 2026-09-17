"""Compile actual solve.cu kernel bodies against a test-only GMP backend."""
from pathlib import Path
import argparse
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--define',action='append',default=[])
parser.add_argument('--source',default='solve.cu')
args = parser.parse_args()
source_path=ROOT/args.source
BUILD=ROOT/({'rns/solve.cu':'build-emulation-rns','quartic/solve.cu':'build-emulation-quartic'}.get(args.source,'build-emulation'))
BUILD.mkdir(exist_ok=True)
source = source_path.read_text()
source = source.replace('#include <cupqc/bigint.hpp>', '#include "tests/cuda_emulation.h"')
source = source.replace('#include <cuda_runtime.h>', '')
# Device launch bounds constrain register allocation only; the CPU adapter
# has no register allocator/SM occupancy model.
source=re.sub(r'__launch_bounds__\([^)]*\)', '', source)
# Resolve the two conditional kernel-declaration branches without preprocessing
# unrelated includes or changing the kernel bodies.
source=re.sub(r'#if FHERMA_MIN_BLOCKS && FHERMA_TPI==1\s*__global__\s*#else\s*__global__\s*#endif\s*void', '__global__ void', source)
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
    body=body.replace('__shfl_xor_sync(', 'co_await emulated_shuffle_xor(')
    decl=source[m.start():start].replace('__global__ void','EmulatedKernel')
    source=source[:m.start()]+decl+body+'co_return;\n}'+source[end:]
source, count = re.subn(
    r'(\w+(?:<[A-Za-z]+(?:,[A-Za-z]+)*>)?)<<<([^,<>]+),\s*([^<>]+)>>>\(([^;]+)\);',
    lambda m: 'emulate_launch('+m[2]+','+m[3].split(',')[0]+',[&] { return '+m[1]+'('+m[4]+'); });', source)
expected={'rns/solve.cu':20,'quartic/solve.cu':22}.get(args.source,14)
assert count == expected, f'Expected {expected} CUDA launch sites, saw {count}; update test adapter explicitly.'
assert '<<<' not in source
(BUILD / 'solve_emulated.cpp').write_text(source)
cmd = ['clang++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-I'+str(ROOT),
       '-DFHERMA_PROFILE=0', '-DFHERMA_GRAPH=0', *['-D'+d for d in args.define],
       '-I/opt/homebrew/include', str(ROOT / 'main.cpp'), str(BUILD / 'solve_emulated.cpp'),
       '-L/opt/homebrew/lib', '-lgmpxx', '-lgmp', '-o', str(BUILD / 'solution')]
subprocess.run(cmd, check=True)
print(BUILD / 'solution')
