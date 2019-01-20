import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
from sklearn.model_selection import train_test_split
from src.split import Split

warnings.filterwarnings(action='once')


class Dataset:
    """
    This class allows a simpler representation of the dataset used
    to build a model in class. It allows loading a remote CSV by
    providing an URL to the initialization method of the object.

        my_data = Dataset(URL)
        
    """
    
    meta = None
    target = None
    features = None
    
    def __init__(self, data_location):
        self.data = pd.read_csv(data_location)
        self.features = list(self.data)
        self.metainfo()
        
    def set_target(self, target):
        if target in self.features:
            self.features.remove(target)
        self.target = target
        self.metainfo()
        
    def metainfo(self):
        """
        Builds metainfromation about the dataset, considering the 
        features that are categorical, numerical or does/doesn't contain NA's.
        """
        meta = dict()
        descr = pd.DataFrame({'dtype': self.data.dtypes, 
                              'NAs': self.data.isna().sum()})
        categorical_features = descr.loc[descr['dtype'] == 'object'].\
            index.values.tolist()
        numerical_features = descr.loc[descr['dtype'] != 'object'].\
            index.values.tolist()
        numerical_features_na = descr.loc[(descr['dtype'] != 'object') & 
                                          (descr['NAs'] > 0)].\
            index.values.tolist()
        categorical_features_na = descr.loc[(descr['dtype'] == 'object') & 
                                            (descr['NAs'] > 0)].\
            index.values.tolist()
        complete_features = descr.loc[descr['NAs'] == 0].index.values.tolist()
        meta['description'] = descr
        meta['all'] = list(self.data)
        meta['features'] = list(self.features)
        meta['target'] = self.target
        meta['categorical'] = categorical_features
        meta['categorical_na'] = categorical_features_na
        meta['numerical'] = numerical_features
        meta['numerical_na'] = numerical_features_na
        meta['complete'] = complete_features
        self.meta = meta
        return self
    
    def describe(self):
        """
        Printout the metadata information collected when calling the 
        metainfo() method.
        """
        if not self.meta:
            self.metainfo()
        print('Available types:', self.meta['description']['dtype'].unique())
        print('{} Features'.format(self.meta['description'].shape[0]))
        print('{} categorical features'.format(
            len(self.meta['categorical'])))
        print('{} numerical features'.format(
            len(self.meta['numerical'])))
        print('{} categorical features with NAs'.format(
            len(self.meta['categorical_na'])))
        print('{} numerical features with NAs'.format(
            len(self.meta['numerical_na'])))
        print('{} Complete features'.format(
            len(self.meta['complete'])))
        print('--')
        print('Target: {}'.format(
            self.target if self.target is not None else 'Not set'))
        
    def select(self, which):
        """
        Returns a subset of the columns of the dataset.
        `which` specifies which subset of features to return
        If it is a list, it returns those feature names in the list,
        And if it is a keywork from: 'all', 'categorical', 'categorical_na',
        'numerical', 'numerical_na', 'complete', 'features', 'target',
        then the list of features is extracted from the metainformation 
        of the dataset.
        """
        assert which in ['all','numerical','categorical','complete',
                         'numerical_na','categorical_na','features',
                         'target']

        if isinstance(which, list):
            return self.data.loc[:, which]
        else:
            return self.data.loc[:, self.meta[which]]
    
    def names(self, which):
        """
        Returns a the names of the columns of the dataset for which the arg
        `which` is specified.
        If it is a list, it returns those feature names in the list,
        And if it is a keywork from: 'all', 'categorical', 'categorical_na',
        'numerical', 'numerical_na', 'complete', then the list of 
        features is extracted from the metainformation of the dataset.
        """
        assert which in ['all','numerical','categorical','complete',
                         'numerical_na','categorical_na']
        return self.meta[which]

    def table(self, which=all, max_width=80):
        """
        Print a tabulated version of the list of elements in a list, using
        a max_width display (default 80).
        """
        assert which in ['all','numerical','categorical','complete',
                         'numerical_na','categorical_na']
        
        f_list = self.names(which)
        if len(f_list) == 0:
            return

        num_features = len(f_list)
        max_length = max([len(feature) for feature in f_list])
        max_fields = int(np.floor(max_width / (max_length+1)))
        col_width = max_length + 1

        print('-'*((max_fields*max_length)+(max_fields-1)))
        for field_idx in range(int(np.ceil(num_features/max_fields))):
            from_idx = field_idx*max_fields
            to_idx = (field_idx*max_fields)+max_fields
            if to_idx > num_features:
                to_idx = num_features
            format_str = ''
            for i in range(to_idx-from_idx):
                format_str += '{{:<{:d}}}'.format(col_width)
            print (format_str.format(*f_list[from_idx:to_idx]))
        print('-'*((max_fields*max_length)+(max_fields-1)))
        
    def outliers(self, which):
        """
        Find outliers, using bonferroni criteria, from the numerical features.
        Returns a list of indices where outliers are present
        'which' can be:
          - 'all': default value
          - 'numerical': only numerical features
          - 'categorical': only categorical features
          - 'complete': only complete features (no NA)
        """
        assert which in ['all','numerical','categorical','complete']
        ols = sm.OLS(endog = self.target, exog = self.select('numerical'))
        fit = ols.fit()
        test = fit.outlier_test()['bonf(p)']
        return list(test[test<1e-3].index) 
    
    def drop_samples(self, index_list):
        """
        Remove the list of samples from the dataset. 
        """
        self.data.drop(self.data.index[index_list])
        
    def replace_na(self, column, value):
        """
        Replace any NA occurrence from the column or list of columns passed 
        by the value passed as second argument.
        """
        if isinstance(column, list) is True:
            for col in column:
                self.data[col].fillna(value, inplace=True)
        else:
            self.data[column].fillna(value, inplace=True)
        self.metainfo()
        
    def split(self,
              seed=1024, 
              test_size=0.2, 
              validation_split=False):
        """
        From an input dataframe, separate features from target, and 
        produce splits (with or without validation).
        """
        assert self.target is not None
        
        X = pd.DataFrame(self.data, columns=self.features)
        Y = pd.DataFrame(self.select('target'))

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, 
            test_size=test_size, random_state=seed)

        if validation_split is True:
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_train, Y_train, 
                test_size=test_size, random_state=seed)
            X_splits = [X_train, X_test, X_val]
            Y_splits = [Y_train, Y_test, Y_val]
        else:
            X_splits = [X_train, X_test]
            Y_splits = [Y_train, Y_test]

        return Split(X_splits), Split(Y_splits)
    