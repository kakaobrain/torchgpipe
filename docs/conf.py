# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'torchgpipe'
copyright = '2019, Kakao Brain'
author = 'Kakao Brain'

# The full version, including alpha/beta/rc tags
about = {}
with open('../torchgpipe/__version__.py') as f:
    exec(f.read(), about)
release = about['__version__']
del about

master_doc = 'index'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # We follow Google style docstrings just like PyTorch.
    'sphinx.ext.napoleon',

    # Allow reference sections using its title.
    'sphinx.ext.autosectionlabel',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Link to PyTorch's documentation.
extensions.append('sphinx.ext.intersphinx')
intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/devdocs/', None),
    'python': ('https://docs.python.org/3', None),
}

# Mock up 'torch' to make sure build on Read the Docs.
autodoc_mock_imports = ['torch']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

html_theme_options = {
    'logo': 'not-pipe.svg',
    'logo_name': True,
    'description': 'GPipe for PyTorch',

    'github_user': 'kakaobrain',
    'github_repo': 'torchgpipe',
    'github_type': 'star',

    'extra_nav_links': {
        'Source Code': 'https://github.com/kakaobrain/torchgpipe',
        'Original Paper': 'https://arxiv.org/abs/1811.06965',
    },
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
