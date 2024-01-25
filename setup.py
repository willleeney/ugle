from setuptools import setup, find_packages

PREREQS = [ "torch==1.12.0" ]

from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

def requires( packages ): 
    from os import system
    from sys import executable as PYTHON_PATH
    from pkg_resources import require
    require( "pip" )
    CMD_TMPLT = '"' + PYTHON_PATH + '" -m pip install %s'
    for pkg in packages: system( CMD_TMPLT % (pkg,) )       

class OrderedInstall( install ):
    def run( self ):
        requires( PREREQS )
        install.run( self )        

class OrderedDevelop( develop ):
    def run( self ):
        requires( PREREQS )
        develop.run( self )        

class OrderedEggInfo( egg_info ):
    def run( self ):
        requires( PREREQS )
        egg_info.run( self )        

CMD_CLASSES = { 
     "install" : OrderedInstall
   , "develop" : OrderedDevelop
   , "egg_info": OrderedEggInfo 
}        

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory /  ".github/README.md").read_text()

setup(name='ugle', 
      description='This is a repository for investigating implementations of GNNs for \
                   unsupervised clustering.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/willleeney/ugle',
      author='William Leeney',
      author_email='will.leeney@outlook.com',
      license_files = ('LICENSE',),
      version='0.6.0', 
      packages=find_packages(),
      install_requires=['pytest',
                        'omegaconf',
                        'torch==1.12.0',
                        'torch_geometric',
                        'torch_scatter',
                        'torch_sparse',
                        'torch_cluster',
                        'torch_spline_conv',
                        'numpy<=1.23.0',
                        'networkx',
                        'gdown',
                        'optuna',
                        'memory_profiler',
                        'line_profiler',
                        'matplotlib',
                        'fast-pytorch-kmeans',
                        'names_generator'],
      cmdclass = CMD_CLASSES,
      python_requires=">=3.9.12",
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
       ' Intended Audience :: Information Technology', 
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],

)
