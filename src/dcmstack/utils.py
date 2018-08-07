"""
Various utils and helpers
"""

import sys, string

# Python 2 / 3 compatibility
# Ideally should just use six module
PY2 = sys.version_info[0] < 3

unicode_str = unicode if PY2 else str
iteritems = (lambda d: d.iteritems()) if PY2 else lambda d: d.items()
ascii_letters = string.letters if PY2 else string.ascii_letters