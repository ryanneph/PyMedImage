"""notifications.py

convenience functions for status notifications
"""
from pushbullet import Pushbullet
import logging

# initialize module logger
logger = logging.getLogger(__name__)

# disable pushbullet library logging
pblogger = logging.getLogger('pushbullet')
pblogger.addHandler(logging.NullHandler())


__PB_API_KEY__ = 'o.6CxHK9sQPjbZApoXwDkHhA4oy0KPD8Po'

# pushbullet config
__pb__ = Pushbullet(__PB_API_KEY__)
__pb_channel_research__ = __pb__.channels[0]

def pushNotification(title, body):
    __pb_channel_research__.push_note(title, body)
