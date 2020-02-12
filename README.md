# PyDataset

Allows a simpler representation of the dataset used to build a model in class. It allows to load a remote CSV by providing an URL to the initialization method of the object, and work on the most common tasks related to data preparation and feature engineering.

To read more about the API, please refer to the [API Documentation Home](https://pydataset.readthedocs.io/en/latest/index.html).

The class `Dataset` still needs more work, but feel free to contribute and experiment with it via pull request.

If you want to use it, I recommend to install via pip with a link to the source, in case you want to update or pull it regularly:

    $ git clone https://github.com/renero/dataset
    $ cd dataset
    $ pip install -e .

or maybe, install it directly from git

    $ pip install git+http://github.com/renero/dataset

Once installed, import it from you python code, normally:

    $ from dataset import Dataset

## Building the PDF/LaTeX versions of the notebooks

I've finally used the extension described in [URL](https://dev.to/alephthoughts/publication-ready-jupyter-notebooks-47ca) to be able to hide some of the code cells.

    $ jupyter nbconvert --to hide_code_pdf Notebook.ipynb

The extension called `hide_code` is installed:

    pip install hide_code
    jupyter nbextension install --py hide_code
    jupyter nbextension enable --py hide_code
    jupyter serverextension enable --py hide_code

And then you simply have to convert the notebook, using `--to hide_code_html`, `--to hide_code_pdf` or `--to hide_code_latexpdf`.

## Build the slides (hiding some code cells)

To get slides showing only the code you want to show, I found a snippet of code from Damian Avila, which has been placed in this folder, called `hide_code_in_slideshow.py`.

To hide some code cells in the presentation, BUT keeping the output, simply, load the `.py` at the beginning of the notebook:

    %run hide_code_in_slideshow.py
    
and then, in those cells you don't want the code to be visible, include a call to the method at the beginning of the cell:

    hide_code_in_slideshow()
    whatever_goes_here()
    ...
    
And it will work when you will run the command:

    jupyter nbconvert --to slides my_notebook.ipynb --post serve
    
