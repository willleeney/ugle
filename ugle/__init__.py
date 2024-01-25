"""
ugle.

This is a repository for investigating implementations of GNNs for unsupervised clustering.
"""
__version__ = "0.6.0"
__author__ = 'William Leeney'

from pkgutil import walk_packages
from importlib import import_module

def import_submodules(package: str, recursive: bool=True):
    """ Import all submodules of a module, recursively, including subpackages

    Args:
        package (str): package (name or actual module)
    Returns:
        results dict[str, ModuleType]: all submodules
    """
    if isinstance(package, str):
        package = import_module(package)
    results = {}
    for loader, name, is_pkg in walk_packages(package.__path__):
        full_name = package.__name__ + '.' + name
        results[full_name] = import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results

import_submodules(__name__)
