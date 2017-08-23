#!/usr/bin/env python

from setuptools import setup, find_packages
from pymedimage._version import VERSION_FULL

setup(name='PyMedImage',
      version=VERSION_FULL,
      description='Medical imaging library for working with Dicom data in research',
      author='Ryan Neph',
      author_email='neph320@gmail.com',
      url='https://github.com/ryanneph/pymedimage',
      packages=find_packages(),
      install_requires=[
      ]
      )
