from PyMerge import merge

inc_dirs = ['../src/benchmark', '../src/chess', '../src/lc0ctl',
            '../src/mcts', '../src/neural/blas', '../src/neural/shared']
src_dirs = []
merge.merge('../src/main.cc', src_dir=src_dirs, inc_dir=inc_dirs, save_full_path='./bot_merged.cpp')