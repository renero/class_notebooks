import warnings

warnings.filterwarnings('ignore')

from dataset import Dataset

print('Pokemon dataset')
pokemon = Dataset('../data/pokemon.csv.gz')
pokemon.set_target('Legendary')
pokemon.describe()
pokemon.summary()

print('--')
hr = Dataset('/Users/renero/Downloads/hr-analytics.zip')
hr.set_target('left')
hr.summary()
print('\n--')
print('List of skewed numerical features:')
print(hr.skewed_features())
print('\n--')
print('Converting categoricals to dummies')
hr.to_categorical(['number_project', 'time_spend_company',
                   'promotion_last_5years', 'Work_accident'])
hr.onehot_encode(hr.names('categorical'))
hr.summary()
print('\nStepwise feature selection')
hr.keep_columns(hr.stepwise_selection()).describe()
