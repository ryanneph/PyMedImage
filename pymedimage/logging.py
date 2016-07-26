"""A collection of logging classes and convenience functions"""

import time

def print_timer(message, time_secs):
    print('{message:s} {time:s}.'.format(
        message=message,
        time=(time.strftime('%H:%M:%S', time.gmtime(time_secs)))
        )
    )


def print_header(title, sep='-'):
    nseps = len(title)
    sep_string = ''.join([sep for i in range(nseps)])
    print('{sep_string:s}\n{title:s}\n{sep_string:s}'.format(title=title, sep_string=sep_string))
    

def print_indent(message, indent=0):
    indent_string = ''.join([' ' for i in range(indent)])
    message = message.replace('\n', '\n' + indent_string)
    print('{indent_string:s}{message:s}'.format(indent_string=indent_string, message=message))

