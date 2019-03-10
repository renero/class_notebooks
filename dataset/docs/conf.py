# -*- coding: utf-8 -*-
import os
import sys


sys.path.insert(0, os.path.abspath('../'))
sys.path.append('/Users/renero/Documents/IE/class_notebooks/dataset/dataset')

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']
source_suffix = '.rst'
master_doc = 'index'
project = u'Dataset'
copyright = u'J. Renero'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = 'default'
autoclass_content = "both"

autodoc_default_options = {
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': 'update, target, numerical, numbers_to_float, \
    meta_tags, meta, features, describe_numerical, describe_categorical,\
    data, categorical_dtypes, all, categorical'
}
