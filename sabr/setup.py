#!/usr/bin/python
# -*- coding: utf-8  -*-

# try:
from setuptools import setup, find_packages
# except ImportError:
#    from distutils.core import setup

config = {
    'name': 'analytics',
    'description': 'Implied calcs, SABR calibration and pricing',
    'author': 'Ale Raj Franco',
    'url': ' ',
    'download_url': 'https://bitbucket.org/incomepro/analytics',
    'author_email': ' ',
    'version': '0.1',
    'install_requires': [  # 'py_lets_be_rational', # for fast implieds
                          'numpy',
                          'pandas',
                          'scipy'],
    'packages': ['analytics'],  # or find_packages()
    'script': [],
}

# i.e. setup('description' = 'My Analytics', 'author' = 'A', ...)
setup(**config)
