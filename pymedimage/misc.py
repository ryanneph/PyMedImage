"""misc.py

collection of miscellanious convenience functions
"""

from itertools import zip_longest
def grouper(n, iterable, fillvalue=None):
    """Unpacks iterables using groupings of n elements

    Ex:  grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def frange(x, y, jump):
    """range generator for floats between x and y with spacing jump"""
    while x < y:
        yield x
        x += jump
