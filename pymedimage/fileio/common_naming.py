import re

# string/path manipulation
def gettype_BRATS17(fname):
    match = re.search(r'_([a-zA-Z0-9]*)\.[\.\w]*$', fname)
    return match.group(1)
