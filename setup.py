#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: arshadzahangirchowdhury
"""

from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='VOC-Net',
    url='https://github.com/arshadzahangirchowdhury/VOC-Net',
    author='M Arshad Zahangir Chowdhury',
    author_email='arshad.zahangir.bd@gmail.com',
    # Needed to actually package something
    packages=find_packages(exclude=['test']),
    # Needed for dependencies
    install_requires=['numpy', 'pandas', 'scipy', 'h5py', 'matplotlib', 'opencv-python',\
                      'scikit-image','scikit-learn', 'seaborn' ,'ipython', 'tensorflow', 'tensorflow'],
    version="1.0",
    license='BSD',
    description='VOC-Net: Volatile Organic Compounds Classification Network',
#     long_description=open('README.md').read(),
)
