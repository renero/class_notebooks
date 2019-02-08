import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

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
