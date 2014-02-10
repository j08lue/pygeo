try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name = 'pygeo',
      version = '0.1.0',
      author = 'Jonas Bluethgen',
      author_email = 'bluthgen@nbi.ku.dk',
      packages = ['pygeo'],
      url = 'http://www.gfy.ku.dk/~bluthgen',
      license = 'LICENSE.txt',
      description = 'Functions for working with geospatial coordinates (lon,lat)',
      long_description = open('README.md').read(),
      )
