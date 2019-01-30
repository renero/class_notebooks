# import nbimporter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.dataset import Dataset
from dython.nominal import associations

houses = Dataset('./data/houseprices_prepared.csv.gz')
houses.describe()

houses.replace_na(column='Electrical', value='Unknown')
houses.replace_na(column=houses.names('categorical_na'), value='None')
houses.set_target('SalePrice')
houses.describe()

houses.drop_columns('Id')
houses.describe()
