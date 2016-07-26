"""A collection of logging classes and convenience functions"""

import time

def printtimer(message, time_secs):
    print('{message:s} {time:s}.'.format(
        message=message,
        time=(time.strftime('%H:%M:%S', time.gmtime(time_secs)))
        )
    )


def print_header(title, sep='-'):
    nseps = len(title)
    sep_string = ''.join([sep for i in range(nseps)])
    print('\n{sep_string:s}\n{title:s}\n{sep_string:s}'.format(title=title, sep_string=sep_string))

