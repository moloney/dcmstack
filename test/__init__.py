"""
Module providing tests for dcmstack
"""
import sys
from os import path

test_dir = path.dirname(__file__)
src_dir = path.normpath(path.join(test_dir, path.pardir, 'src'))

if sys.path[0] != src_dir:
    sys.path.insert(0, src_dir)
