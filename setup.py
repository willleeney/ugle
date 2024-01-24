from setuptools import setup, find_packages

setup(name='ugle', 
      description='This is a repository for investigating implementations of GNNs for \
                   unsupervised clustering.',
      url='https://github.com/willleeney/ugle',
      author='William Leeney',
      author_email='will.leeney@outlook.com',
      license_files = ('LICENSE',),
      version='0.1.0', 
      packages=find_packages(),
      install_requires=['pytest',
                        'omegaconf',
                        'torch',
                        'torch_geometric',
                        'torch_scatter',
                        'torch_sparse',
                        'torch_cluster',
                        'torch_spline_conv',
                        'numpy<=1.23.0',
                        'gdown',
                        'optuna',
                        'memory_profiler',
                        'line_profiler',
                        'matplotlib',
                        'fast-pytorch-kmeans',
                        'names_generator'],
      python_requires=">=3.9.12",
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
       ' Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],

)

