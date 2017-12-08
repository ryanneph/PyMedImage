import os
import re

# string/path manipulation
def matchtype(fname, type, typegetter=None):
    if callable(typegetter):
        return typegetter(fname) == type
    else:
        return type in fname

def sanitize(string, dirty_chars=['.']):
    for c in dirty_chars:
        string = string.replace(c, '\{}'.format(c))
    return string

def getFileType(fname):
    m = re.search(r'\.[a-zA-Z0-9]*\.?[a-zA-Z0-9]*$', fname)
    if m is not None: return m.group(0)
    else: return None

def isFileByExt(fname, exts=None):
    if not exts: return True
    if isinstance(exts, str): exts=[exts]
    for e in [sanitize(x) for x in exts]:
        if re.search(r'{}$'.format(e), fname, re.IGNORECASE) is not None:
            return True
    return False

