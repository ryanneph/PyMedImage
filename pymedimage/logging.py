"""A collection of logging classes and convenience functions"""

import time

# global indent settings
g_indents = {1: 0,
             2: 2,
             3: 4,
             4: 6 }

def __get_indent_string(indent):
    """Contructs proper number of indent spaces"""
    return ''.join([' ' for i in range(indent)])


def print_timer(message, time_secs, indent=0):
    if (indent>0):
        message = ''.join(__get_indent_string(indent), message)
    print('{message:s} {time:s}'.format(
        message=message,
        time=(time.strftime('%H:%M:%S', time.gmtime(time_secs)))
        )
    )


def print_header(title, sep='-'):
    nseps = len(title)
    sep_string =  ''.join([sep for i in range(nseps)])
    print('{sep_string:s}\n{title:s}\n{sep_string:s}'.format(title=title, sep_string=sep_string))
    

def print_indent(message, indent=0):
    indent_string = __get_indent_string(indent)
    message = message.replace('\n', '\n' + indent_string)
    print('{indent_string:s}{message:s}'.format(indent_string=indent_string, message=message))

