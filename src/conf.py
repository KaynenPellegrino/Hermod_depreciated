import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
project = 'Hermod'
copyright = '2024, Kaynen Pellegrino'
author = 'Kaynen Pellegrino'
release = '0.1'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints']
napoleon_google_docstring = True
napoleon_numpy_docstring = False
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'alabaster'
html_static_path = ['_static']
