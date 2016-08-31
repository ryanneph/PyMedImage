"""misc.py

collection of miscellanious convenience functions
"""

import time
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



# global indent settings
g_indents = {1: 0,
             2: 2,
             3: 4,
             4: 6 }

def __get_indent_string(indent):
    """Contructs proper number of indent spaces"""
    return ''.join([' ' for i in range(indent)])

def timer(message, time_secs, indent=0):
    if (indent>0):
        message = ''.join([__get_indent_string(indent), message])
    return '{message:s} {time:s}'.format(message=message,
                                         time=time.strftime('%H:%M:%S', time.gmtime(time_secs)) )

def header(title, sep='*'):
   nseps = 3
   sep_string =  ''.join([sep for i in range(nseps)])
   return '{sep_string:s} {title:s} {sep_string:s}'.format(title=title, sep_string=sep_string)


def headerBlock(title, sep='-'):
   nseps = len(title)
   sep_string =  ''.join([sep for i in range(nseps)])
   return '\n{sep_string:s}\n{title:s}\n{sep_string:s}'.format(title=title, sep_string=sep_string)

def indent(message, indent=0):
    indent_string = __get_indent_string(indent)
    message = message.replace('\n', '\n' + indent_string)
    return '{indent_string:s}{message:s}'.format(indent_string=indent_string, message=message)
