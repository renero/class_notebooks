.. dataset documentation master file, created by
   sphinx-quickstart on Sat Mar  9 09:47:15 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Dataset's documentation!
===================================

.. currentmodule:: dataset.dataset

Allows a simpler representation of the dataset used to build a model in class.
It allows to load a remote CSV by providing an URL to the
initialization method of the object, and
work on the most common tasks related to data preparation and
feature engineering.::

      >>> my_data = Dataset(URL)

or, if you prefer to create from an existing Pandas DataFrame::

      >>> my_data = Dataset.from_dataframe(my_dataframe)


Contents:

.. toctree::
  :maxdepth: 1

  dataset
  modules
