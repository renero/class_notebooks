import warnings

warnings.filterwarnings('ignore')

from dataset import Dataset

print('Pokemon dataset')
pokemon = Dataset('../data/pokemon.csv.gz')
pokemon.set_target('Legendary')
pokemon.describe()
pokemon.summary()

print('--')
hr = Dataset('../data/hr-analytics.zip')
hr.set_target('left')
hr.summary('numerical')
print('\n--\nList of skewed numerical features:')
print(hr.skewed_features())

print('\n--\nConverting categoricals to dummies')
hr.to_categorical(['number_project', 'time_spend_company',
                   'promotion_last_5years', 'Work_accident'])
hr.onehot_encode(hr.names('categorical'))
hr.summary()

print('\nStepwise feature selection')
hr.keep_columns(hr.stepwise_selection()).describe()

print('\nFeatures highly correlated')
hr.drop_columns(hr.correlated()).describe()

print('\nOutliers')
lof_outliers = hr.outliers()

# La clave ahora es coger el percentil 1% inferior, que recoge los outliers
# para ello, tengo que conseguir el indice de esos elementos para poder
# eliminarlos