import __future__
from setuptools import setup
import os

# Authors : David C.C. Lu <davidlu89@gmail.com>
#
# License : MIT

setup(
    name = "mspacman",
    version = 0.1,
    packages = ['mspacman', 'mspacman.viz', 'mspacman.algorithm', 'mspacman.generator', 'mspacman.app', 'mspacman.utilities'],
    install_requires=['numpy', 'scipy', 'pytf', 'matplotlib'],
    author = "David Lu",
    author_email = "davidlu89@gmail.com",
    description = "mspacman is a tool for analyzing Phase-Amplitude Coupling",
    url='https://github.com/davidlu89/mspacman',
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'],
        platforms='any'
)
