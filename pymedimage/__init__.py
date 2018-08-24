import os
import warnings

#  # add root to PATH
#  import sys
#  sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
#  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from . import ( rttypes, features, features_gpu, calculate_features, loggers, visual, visualgui )

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(NullHandler())

# enable all warnings within the lib
warnings.filterwarnings('default', module='pymedimage.*', category=DeprecationWarning)
warnings.filterwarnings('default', module='pymedimage.*', category=PendingDeprecationWarning)
