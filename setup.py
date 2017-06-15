#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='attorch',
    version='0.0.0',
    description='Torch utilities used in the Tolias lab @ Baylor College of Medicine',
    author='Fabian Sinz',
    author_email='sinz@bcm.edu',
    url='https://github.com/atlab/attorch',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm','gitpython','python-twitter','scikit-image', 'datajoint', 'atflow', 'torch'],
)
