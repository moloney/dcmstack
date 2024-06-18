"""
Various utils and helpers
"""

import sys, string, traceback, pdb

# Python 2 / 3 compatibility
# Ideally should just use six module
PY2 = sys.version_info[0] < 3

unicode_str = unicode if PY2 else str
byte_str = str if PY2 else bytes
str_types = (unicode_str, byte_str)
iteritems = (lambda d: d.iteritems()) if PY2 else lambda d: d.items()
ascii_letters = string.letters if PY2 else string.ascii_letters


def pdb_except_hook(type, value, tb):
    """Function that can be used as excepthook to start PDB"""
    traceback.print_exception(type, value, tb)
    print()
    pdb.post_mortem(tb)
