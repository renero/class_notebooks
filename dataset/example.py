import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from dataset import Dataset


pokemon = Dataset('../data/pokemon.csv.gz')
pokemon.set_target('Legendary')
pokemon.describe()
pokemon.summary()
