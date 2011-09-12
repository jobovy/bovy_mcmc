from setuptools import setup #, Extension
import os, os.path
import re

longDescription= ""


setup(name='bovy_mcmc',
      version='1.',
      description='MCMC routines',
      author='Jo Bovy',
      author_email='bovy@ias.edu',
      license='New BSD',
      long_description=longDescription,
      url='https://github.com/jobovy/bovy_mcmc',
      package_dir = {'bovy_mcmc/': ''},
      packages=['bovy_mcmc'],
      dependency_links = ['https://github.com/dfm/MarkovPy/tarball/master#egg=MarkovPy'],
      install_requires=['MarkovPy']
      )
