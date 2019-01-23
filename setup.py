"""
Copyright (C) 2018  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
from distutils.core import setup

setup(
    name='perfect_match',
    version='1.0.0',
    packages=['perfect_match', 'perfect_match.apps',
              'perfect_match.data_access',
              'perfect_match.data_access.ihdp', 'perfect_match.data_access.jobs',
              'perfect_match.data_access.news', 'perfect_match.data_access.tcga', 'perfect_match.data_access.twins',
              'perfect_match.models',
              'perfect_match.models.baselines', 'perfect_match.models.baselines.cfr', 'perfect_match.models.baselines.ganite_package',
              'perfect_match.models.benchmarks'],
    url='schwabpatrick.com',
    author='Patrick Schwab',
    author_email='patrick.schwab@hest.ethz.ch',
    license=open('LICENSE.txt').read(),
    long_description=open('README.md').read(),
    install_requires=[
        "Keras >= 1.2.2",
        "tensorflow == 1.4.0",
        "matplotlib >= 1.3.1",
        "pandas >= 0.18.0",
        "h5py >= 2.6.0",
        "scikit-learn == 0.19.0",
        "numpy >= 1.14.1",
        "rpy2 == 2.8.6"
    ]
)
