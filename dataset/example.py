import warnings

warnings.filterwarnings('ignore')

import pandas as pd
from dataset import Dataset


houses = Dataset('../data/houseprices_prepared.csv.gz')
houses.describe()

houses.replace_na(column='Electrical', value='Unknown')
houses.replace_na(column=houses.names('categorical_na'), value='None')
houses.set_target('SalePrice')
houses.describe()

houses.drop_columns('Id')
houses.describe()

houses.aggregate(['1stFlrSF','2ndFlrSF','BsmtFinSF1','BsmtFinSF2'], 'HouseSF')
houses.describe()

houses.aggregate(['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch',
                  'WoodDeckSF'], 'HousePorch')
houses.describe()

houses.aggregate(['FullBath', 'BsmtFullBath', 'HalfBath', 'BsmtHalfBath'],
                 'HouseBaths')
houses.describe()

#
# Categorical examples
#
d = {'col1': [1, 2, 3], 'col2': ['a', 'a', 'b']}
df = pd.DataFrame(data=d)
ds = Dataset.from_dataframe(df)
ds.set_target('col2')
ds.describe()
print(list(ds.select('numerical')))