import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import warnings

from sklearn.model_selection import train_test_split
from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

from src.split import Split
from src.correlations import cramers_v, theils_u


warnings.filterwarnings(action='once')

#
# Correlation ideas taken from:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#

class Dataset:
    """
    This class allows a simpler representation of the dataset used
    to build a model in class. It allows loading a remote CSV by
    providing an URL to the initialization method of the object.

        my_data = Dataset(URL)
        
    """
    
    meta = None
    data = None
    target = None
    features = None

    meta_tags = ['all', 'numerical', 'categorical', 'complete',
                 'numerical_na', 'categorical_na', 'features', 'target']

    
    def __init__(self, data_location):
        self.data = pd.read_csv(data_location)
        self.features = self.data
        self.metainfo()
        
    def set_target(self, target_name):
        """
        Set the target variable for this dataset. This will create a new
        property of the object called 'target' that will contain the 
        target column of the dataset, and that column will be removed
        from the list of features.
        Example:
        
            my_data.set_target('SalePrice')
            
        """
        if target_name in list(self.features):
            self.target = self.features.loc[:, target_name].copy()
            self.features.drop(self.target.name, axis=1, inplace=True)
        else:
            self.target = self.data.loc[:, target_name].copy()
        self.metainfo()
        
    def metainfo(self):
        """
        Builds metainfromation about the dataset, considering the 
        features that are categorical, numerical or does/doesn't contain NA's.
        """
        meta = dict()
        
        # Build the subsets per data ype (list of names)
        descr = pd.DataFrame({'dtype': self.features.dtypes, 
                              'NAs': self.features.isna().sum()})
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
        
        # Update META-information
        meta['description'] = descr
        meta['all'] = list(self.data)
        meta['features'] = list(self.features)
        meta['target'] = self.target.name if self.target is not None else None
        meta['categorical'] = categorical_features
        meta['categorical_na'] = categorical_features_na
        meta['numerical'] = numerical_features
        meta['numerical_na'] = numerical_features_na
        meta['complete'] = complete_features
        self.meta = meta
        return self
    
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
        if isinstance(which, list):
            return self.data.loc[:, which]
        else:
            assert which in self.meta_tags
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
        assert which in self.meta_tags
        return self.meta[which]

    def outliers(self):
        """
        Find outliers, using bonferroni criteria, from the numerical features.
        Returns a list of indices where outliers are present
        """
        ols = sm.OLS(endog = self.target, exog = self.select('numerical'))
        fit = ols.fit()
        test = fit.outlier_test()['bonf(p)']
        return list(test[test<1e-3].index)
    
    def skewness(self, threshold=0.75, fix=False, return_series=False):
        """
        Returns the list of numerical features that present skewness
        :return: A pandas Series with the features and their skewness
        """
        df = self.select('numerical')
        feature_skew = df.apply(
            lambda x: skew(x)).sort_values(ascending=False)

        if fix is True:
            high_skew = feature_skew[feature_skew > threshold]
            skew_index = high_skew.index
            for feature in skew_index:
                self.features[feature] = boxcox1p(
                    df[feature], boxcox_normmax(df[feature] + 1))
        if return_series is True:
            return feature_skew

    def numerical_correlated(self,
                             method='spearman',
                             threshold=0.9):
        """
        Build a correlation matrix between all the features in data set
        :param subset: Specify which subset of features use to build the
        correlation matrix. Default 'features'
        :param method: Method used to build the correlation matrix.
        Default is 'Spearman' (Other options: 'Pearson')
        :param threshold: Threshold beyond which considering high correlation.
        Default is 0.9
        :return: The list of columns that are highly correlated and could be
        droped out from dataset.
        """
        corr_matrix = np.absolute(
            self.select('numerical').corr(method=method)).abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than threshold
        return [column for column in upper.columns
                   if any(abs(upper[column]) > threshold)], corr_matrix

    def categorical_correlation(self, threshold=0.9):
        """
        Generates a correlation matrix for the categorical variables in dataset
        :param method: 'cramers_v' or 'theils_u'
        :param threshold: Limit from which correlations is considered high.
        :return: the list of categorical variables with HIGH correlation and
        the correlation matrix
        """
        columns = self.meta['categorical']
        corr = pd.DataFrame(index=columns, columns=columns)
        for i in range(0, len(columns)):
            for j in range(i, len(columns)):
                if i == j:
                    corr[columns[i]][columns[j]] = 1.0
                else:
                    cell = cramers_v(self.features[columns[i]],
                                     self.features[columns[j]])
                    corr[columns[i]][columns[j]] = cell
                    corr[columns[j]][columns[i]] = cell
        corr.fillna(value=np.nan, inplace=True)
        # Select upper triangle of correlation matrix
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than threshold
        return [column for column in upper.columns
                   if any(abs(upper[column]) > threshold)], corr

    def under_represented_features(self, threshold=0.98):
        """
        Returns the list of categorical features with unrepresented categories
        or a clear unbalance between the values that can take.
        :param threshold: The upper limit of the most represented category
        of the feature.
        :return: the list of features that with unrepresented categories.
        """
        under_rep = []
        for column in self.meta['categorical']:
            counts = self.features[column].value_counts()
            majority_freq = counts.iloc[0]
            if (majority_freq / len(self.features)) > threshold:
                under_rep.append(column)
        return under_rep

    def drop_columns(self, columns_list):
        """
        Drop one or a list of columns from the dataset.
        Example:
        
            my_data.drop_columns('column_name')
            my_data.drop_columns(['column1', 'column2', 'column3'])
        """
        if isinstance(columns_list, list) is not True:
            columns_list = [columns_list]
        for column in columns_list:
            if column in self.names('features'):
                self.features.drop(column, axis=1, inplace=True)
        self.metainfo()
    
    def drop_samples(self, index_list):
        """
        Remove the list of samples from the dataset. 
        """
        self.data.drop(self.data.index[index_list])
        self.metainfo()
        
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

    def describe(self):
        """
        Printout the metadata information collected when calling the
        metainfo() method.
        """
        if self.meta is None:
            self.metainfo()

        print('\nAvailable types:', self.meta['description']['dtype'].unique())
        print('{} Features'.format(len(self.meta['features'])))
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
            self.meta['target'] if self.target is not None else 'Not set'))

    def table(self, which=all, max_width=80):
        """
        Print a tabulated version of the list of elements in a list, using
        a max_width display (default 80).
        """
        assert which in self.meta_tags

        f_list = self.names(which)
        if len(f_list) == 0:
            return

        num_features = len(f_list)
        max_length = max([len(feature) for feature in f_list])
        max_fields = int(np.floor(max_width / (max_length + 1)))
        col_width = max_length + 1

        print('-' * ((max_fields * max_length) + (max_fields - 1)))
        for field_idx in range(int(np.ceil(num_features / max_fields))):
            from_idx = field_idx * max_fields
            to_idx = (field_idx * max_fields) + max_fields
            if to_idx > num_features:
                to_idx = num_features
            format_str = ''
            for i in range(to_idx - from_idx):
                format_str += '{{:<{:d}}}'.format(col_width)
            print(format_str.format(*f_list[from_idx:to_idx]))
        print('-' * ((max_fields * max_length) + (max_fields - 1)))

    def plot_corr_matrix(self, corr_matrix):
        f, ax = plt.subplots(figsize=(11, 9))
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=0.75, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5});
        plt.show();