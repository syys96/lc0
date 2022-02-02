# from PyMerge import merge
import sys
sys.path.append('/Users/syys/PycharmProjects/PyMerge')
from PyMerge.merge import merge as pm

inc_dirs = ['../src/benchmark', '../src/chess', '../src/lc0ctl',
            '../src/mcts', '../src/neural/blas', '../src/neural/shared',
            '.', './meson_conf']
src_dirs = []
not_def_macro = ['_MSC_VER', '_WIN32']
pm('../src/main.cc', src_dir=src_dirs, inc_dir=inc_dirs, save_full_path='./bot_merged_v2.cpp',
            extra_hea_suf=['.inc'], extra_src_suf=['.cc'], pro_root_dir='../src', not_def_macro=not_def_macro)