import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.special import boxcox1p
from scipy.stats import skew, boxcox_normmax
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn_pandas import DataFrameMapper

from dataset.correlations import cramers_v
from dataset.split import Split

warnings.simplefilter(action='ignore')

#
# Correlation ideas taken from:
# https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
#


class Dataset(object):
    """
    This class allows a simpler representation of the dataset used
    to build a model in class. It allows loading a remote CSV by
    providing an URL to the initialization method of the object.

        my_data = Dataset(URL)

        my_data = Dataset.from_dataframe(my_dataframe)
        
    """
    
    meta = None
    data = None
    target = None
    features = None
    numerical = None
    categorical = None

    meta_tags = ['all', 'numerical', 'categorical', 'complete',
                 'numerical_na', 'categorical_na', 'features', 'target']
    categorical_dtypes = ['bool', 'object', 'string']

    def __init__(self, data_location=None, data_frame=None, *args, **kwargs):
        """
        Wrapper over the method read_csv from pandas, so you can user variadic
        arguments, as if you were using the actual read_csv
        :param data_location: path or url to the file
        :param data_frame: in case this method is called from the class method
        this parameter is passing the actual dataframe to read data from
        :param args: variadic unnamed arguments to pass to read_csv
        :param kwargs: variadic named arguments to pass to read_csv
        """
        if data_location is not None:
            self.features = pd.read_csv(data_location, *args, **kwargs)
        else:
            if data_frame is not None:
                self.features = data_frame
            else:
                raise RuntimeError(
                    "No data location, nor DataFrame passed to constructor")
        self.numbers_to_float()
        self.metainfo()

    @classmethod
    def from_dataframe(cls, df):
        return cls(data_location=None, data_frame=df)
        
    def numbers_to_float(self):
        columns = self.features.select_dtypes(include=[np.number]).columns.tolist()
        for column_name in columns:
            self.features[column_name] = pd.to_numeric(
                self.features[column_name]).astype(float)
        return

    def set_target(self, target_name):
        """
        Set the target variable for this dataset. This will create a new
        property of the object called 'target' that will contain the 
        target column of the dataset, and that column will be removed
        from the list of features.
        Example:
        
            my_data.set_target('SalePrice')
            
        """
        assert target_name in list(self.features), "Target name NOT recognized"

        self.target = self.features.loc[:, target_name].copy()
        self.features.drop(target_name, axis=1, inplace=True)
        self.metainfo()
        return self
        
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
        meta['all'] = list(self.features)
        meta['features'] = list(self.features)
        meta['target'] = self.target.name if self.target is not None else None
        meta['categorical'] = categorical_features
        meta['categorical_na'] = categorical_features_na
        meta['numerical'] = numerical_features
        meta['numerical_na'] = numerical_features_na
        meta['complete'] = complete_features
        self.meta = meta

        # Update macro access properties
        self.numerical = self.select('numerical')
        self.categorical = self.select('categorical')
        return self
    
    def outliers(self):
        """
        Find outliers, using bonferroni criteria, from the numerical features.
        Returns a list of indices where outliers are present
        """
        # ols = sm.OLS(endog=self.target, exog=self.select('numerical'))
        # fit = ols.fit()
        # test = fit.outlier_test()['bonf(p)']
        # return list(# test[test < 1e-3].index)
        X = self.select('numerical')
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        y_pred = lof.fit_predict(X)
        print(lof.negative_outlier_factor_)
        return lof.negative_outlier_factor_


    def scale(self, features_of_type='numerical', return_series=False):
        """
        Scales numerical features in the dataset, unless the parameter 'what'
        specifies any other subset selection primitive.
        :param features_of_type: Subset selection primitive
        :return: the subset scaled.
        """
        assert features_of_type in self.meta_tags
        subset = self.select(features_of_type)
        mapper = DataFrameMapper([(subset.columns, StandardScaler())])
        scaled_features = mapper.fit_transform(subset.copy())
        self.features[self.names(features_of_type)] = pd.DataFrame(
            scaled_features,
            index=subset.index,
            columns=subset.columns)
        self.metainfo()
        if return_series is True:
            return self.features[self.names(features_of_type)]
        else:
            return self

    def fix_skewness(self,
                         features_of_type='numerical',
                         return_series=False):
        """
        Ensures that the numerical features in the dataset, unless the
        parameter 'what' specifies any other subset selection primitive,
        fit into a normal distribution by applying the Yeo-Johnson transform
        :param features_of_type: Subset selection primitive
        :param return_series: Return the normalized series
        :return: the subset fitted to normal distribution.
        """
        assert features_of_type in self.meta_tags
        subset = self.select(features_of_type)
        mapper = DataFrameMapper([(subset.columns, PowerTransformer(
            method='yeo-johnson',
            standardize=False))])
        normed_features = mapper.fit_transform(subset.copy())
        self.features[self.names(features_of_type)] = pd.DataFrame(
            normed_features,
            index=subset.index,
            columns=subset.columns)
        self.metainfo()
        if return_series is True:
            return self.features[self.names(features_of_type)]
    
    def skewed_features(self, threshold=0.75, fix=False, return_series=False):
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

    def correlated(self, threshold=0.9):
        """
        Return the features that are highly correlated to with other
        variables, either numerical or categorical, based on the threshold. For
        numerical variables Spearman correlation is used, for categorical
        cramers_v
        :param threshold: correlation limit above which features are considered
                          highly correlated.
        :return: the list of features that are highly correlated, and should be
                 safe to remove.
        """
        corr_categoricals, _ = self.categorical_correlated(threshold)
        corr_numericals, _ = self.numerical_correlated(threshold)
        return corr_categoricals + corr_numericals

    def numerical_correlated(self, threshold=0.9):
        """
        Build a correlation matrix between all the features in data set
        :param threshold: Threshold beyond which considering high correlation.
        Default is 0.9
        :return: The list of columns that are highly correlated and could be
        drop out from dataset.
        """
        corr_matrix = np.absolute(
            self.select('numerical').corr(method='spearman')).abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than threshold
        return [column for column in upper.columns
                   if any(abs(upper[column]) > threshold)], corr_matrix

    def categorical_correlated(self, threshold=0.9):
        """
        Generates a correlation matrix for the categorical variables in dataset
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

    def stepwise_selection(self,
                           initial_list=None,
                           threshold_in=0.01,
                           threshold_out=0.05,
                           verbose=False):
        """
        Perform a forward-backward feature selection based on p-value from
        statsmodels.api.OLS
        Your features must be all numerical, so be sure to onehot_encode them
        before calling this method.
        Always set threshold_in < threshold_out to avoid infinite looping.
        All features involved must be numerical and types must be float.
        Target variable must also be float. You can convert it back to a
        categorical type after calling this method.

        :parameter initial_list: list of features to start with (column names
        of X)
        :parameter threshold_in: include a feature if its p-value < threshold_in
        :parameter threshold_out: exclude a feature if its
        p-value > threshold_out
        :parameter verbose: whether to print the sequence of inclusions and
        exclusions
        :return: list of selected features

        Example:

            my_data.stepwise_selection()

        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        Taken from: https://datascience.stackexchange.com/a/24823
        """
        if initial_list is None:
            initial_list = []
        assert len(self.names('categorical')) == 0
        assert self.target.dtype.name == 'float64'

        included = list(initial_list)
        while True:
            changed = False
            # forward step
            excluded = list(set(self.features.columns) - set(included))
            new_pval = pd.Series(index=excluded)
            for new_column in excluded:
                model = sm.OLS(self.target, sm.add_constant(
                    pd.DataFrame(self.features[included + [new_column]]))).fit()
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed = True
                if verbose:
                    print('Add  {:30} with p-value {:.6}'.format(best_feature,
                                                                 best_pval))
            # backward step
            model = sm.OLS(self.target, sm.add_constant(
                pd.DataFrame(self.features[included]))).fit()
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max()  # null if p-values is empty
            if worst_pval > threshold_out:
                changed = True
                worst_feature = pvalues.argmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature,
                                                                 worst_pval))
            if not changed:
                break
        return included

    #
    # Methods are related to data manipulation of the pandas dataframe.
    #

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
            return self.features.loc[:, which]
        else:
            assert which in self.meta_tags
            return self.features.loc[:, self.meta[which]]

    def names(self, which='all'):
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

    def onehot_encode(self, to_convert=None):
        """
        Encodes the categorical features in the dataset, with OneHotEncode

        :parameter to_convert: column or list of columns to be one-hot encoded.
        The only restriction is that the target variable cannot be specified
        in the list of columns and therefore, cannot be onehot encoded.

        Example:

            # Encodes a single column named 'my_column_name'
            my_data.onehot_encode('my_column_name')

            # Encodes 'col1' and 'col2'
            my_data.onehot_encode(['col1', 'col2'])

            # Encodes all categorical features in the dataset
            my_data.onehot_encode(my_data.names('categorical'))

        """
        assert to_convert is not None
        if isinstance(to_convert, list) is not True:
            to_convert = [to_convert]

        new_df = self.features[
            self.features.columns.difference(to_convert)].copy()
        for column_to_convert in to_convert:
            new_df = pd.concat(
                [new_df,
                 pd.get_dummies(
                     self.features[column_to_convert],
                     prefix=column_to_convert,
                     dtype=float)
                 ],
                axis=1)
        self.features = new_df.copy()
        self.metainfo()
        return self

    def add_column(self, serie):
        """
        Add a Series as a new column to the dataset.
        Example:

            my_data.add_column(serie)
            my_data.add_column(name=pandas.Series().values)
        """
        if serie.name not in self.names('features'):
            self.features[serie.name] = serie.values
            self.metainfo()
        return self

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
        return self

    def keep_columns(self, to_keep):
        """
        Keep only one or a list of columns from the dataset.
        Example:

            my_data.keep_columns('column_name')
            my_data.keep_columns(['column1', 'column2', 'column3'])
        """
        if isinstance(to_keep, list) is not True:
            to_keep = [to_keep]
        to_drop = list(set(list(self.features)) - set(to_keep))
        self.drop_columns(to_drop)
        return self

    def aggregate(self,
                  col_list,
                  new_column,
                  operation='sum',
                  drop_columns=True):
        """
        Perform an arithmetic operation on the given columns, and places the
        result on a new column, removing the original ones.

        Example: if we want to sum the values of column1 and column2 into a
        new column called 'column3', we use:

            my_data.aggregate(['column1', 'column2'], 'column3')

        As a result, 'my_data' will remove 'column1' and 'column2', and the
        operation will be the sum of the values, as it is the default operation.

        :param col_list: the list of columns over which the operation is done
        :param new_column: the name of the new column to be generated from the
        operation
        :param drop_columns: whether remove the columns used to perfrom the
        aggregation
        :param operation: the operation to be done over the column values for
        each row. Examples: 'sum', 'diff', 'max', etc. By default, the operation
        is the sum of the values.
        :return: the Dataset object
        """
        assert operation in dir(type(self.features))
        for col_name in col_list:
            assert col_name in list(self.features)
        self.features[new_column] = getattr(
            self.features[col_list],
            operation)(axis=1)
        if drop_columns is True:
            self.drop_columns(col_list)
        else:
            self.metainfo()
        return self

    def drop_samples(self, index_list):
        """
        Remove the list of samples from the dataset. 
        """
        self.features = self.features.drop(self.features.index[index_list])
        if self.target is not None:
            self.target = self.target.drop(self.target.index[index_list])
        self.metainfo()
        return self
        
    def nas(self):
        """
        Returns the list of features that present NA entries
        :return: the list of feature names presenting NA
        """
        return self.names('numerical_na') + self.names('categorical_na')

    def replace_na(self, column, value):
        """
        Replace any NA occurrence from the column or list of columns passed
        by the value passed as second argument.
        :param column: Column name or list of column names from which to
        replace NAs with the value passes in the second argument
        :param value: value to be used as replacement
        :return: the object.
        """
        if isinstance(column, list) is True:
            for col in column:
                self.features[col].fillna(value, inplace=True)
        else:
            self.features[column].fillna(value, inplace=True)
        self.metainfo()
        return self

    def drop_na(self):
        """
        Drop samples with NAs from the features. If any value is infinite
        or -infinite, it is converted to NA, and removed also.

        :return: object
        """
        self.features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        self.features.dropna()
        self.metainfo()
        return self
        
    def split(self,
              seed=1024, 
              test_size=0.2, 
              validation_split=False):
        """
        From an input data frame, separate features from target, and
        produce splits (with or without validation).
        """
        assert self.target is not None
        
        x = pd.DataFrame(self.features, columns=self.names('features'))
        y = pd.DataFrame(self.target)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=test_size, random_state=seed)

        if validation_split is True:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train,
                test_size=test_size, random_state=seed)
            x_splits = [x_train, x_test, x_val]
            y_splits = [y_train, y_test, y_val]
        else:
            x_splits = [x_train, x_test]
            y_splits = [y_train, y_test]

        return Split(x_splits), Split(y_splits)

    def to_numerical(self, to_convert):
        """
        Convert the specified column or columns to numbers
        :param to_convert: column or column list to be converted
        :return: object
        """
        if isinstance(to_convert, list) is not True:
            to_convert = [to_convert]

        for column_name in to_convert:
            if column_name in list(self.features):
                self.features[column_name] = pd.to_numeric(
                    self.features[column_name])
            else:
                self.target = pd.to_numeric(self.target)

        self.metainfo()
        return self

    def to_categorical(self, to_convert):
        """
        Convert the specified column or columns to categories
        :param to_convert: column or column list to be converted
        :return: object
        """
        if isinstance(to_convert, list) is not True:
            to_convert = [to_convert]

        for column_name in to_convert:
            if column_name in list(self.features):
                self.features[column_name] = self.features[column_name].apply(str)
            else:
                self.target = self.target.apply(str)

        self.metainfo()
        return self

    #
    # Description methods, printing out summaries for dataset or features.
    #

    def describe_dataset(self):
        """
        Printout the metadata information collected when calling the
        metainfo() method.
        """
        if self.meta is None:
            self.metainfo()

        print('{} Features. {} Samples'.format(
            len(self.meta['features']), self.features.shape[0]))
        print('Available types:', self.meta['description']['dtype'].unique())
        print('  · {} categorical features'.format(
            len(self.meta['categorical'])))
        print('  · {} numerical features'.format(
            len(self.meta['numerical'])))
        print('  · {} categorical features with NAs'.format(
            len(self.meta['categorical_na'])))
        print('  · {} numerical features with NAs'.format(
            len(self.meta['numerical_na'])))
        print('  · {} Complete features'.format(
            len(self.meta['complete'])))
        print('--')
        if self.target is not None:
            print('Target: {} ({})'.format(
                self.meta['target'], self.target.dtype.name))
            if self.target.dtype.name == 'object':
                self.describe_categorical(self.target)
            else:
                self.describe_numerical(self.target)
        else:
            print('Target: Not set')
        return

    def describe_categorical(self, feature, inline=False):
        """
        Describe a categorical column by printing num classes and proportion
        :return: nothing
        """
        num_categories = feature.nunique()
        cat_names = feature.unique()
        cat_counts = feature.value_counts().values
        cat_proportion = [count / cat_counts.sum()
                          for count in cat_counts]
        if inline is False:
            print('\'', feature.name, '\' (', feature.dtype.name, ')', sep='')
            print('  {} categories'.format(num_categories))
            for cat in range(len(cat_proportion)):
                print('  · \'{}\': {} ({:.04})'.format(
                    cat_names[cat], cat_counts[cat], cat_proportion[cat]))
        else:
            if num_categories <= 4:
                max_categories = num_categories
                trail = ''
            else:
                max_categories = 4
                trail = '...'
            header = '{:d} categs. '.format(num_categories)
            body = '\'{}\'({:d}, {:.4f}) ' * max_categories
            values = [(cat_names[cat], cat_counts[cat], cat_proportion[cat])
                for cat in range(max_categories)]
            values_flattened = list(sum(values, ()))
            body_formatted = body.format(*values_flattened)
            return header + body_formatted + trail


    def numerical_description(self, feature):
        """
        Build a dictionary with the main numerical descriptors for a feature.
        :param feature: The feature (column) to be analyzed
        :return: a dictionary with the indicators and its values.
        """
        description = dict()
        description['Min.'] = np.min(feature)
        description['1stQ'] = np.percentile(feature, 25)
        description['Med.'] = np.median(feature)
        description['Mean'] = np.mean(feature)
        description['3rdQ'] = np.percentile(feature, 75)
        description['Max.'] = np.max(feature)
        return description

    def describe_numerical(self, feature, inline=False):
        """
        Describe a numerical column by printing min, max, med, mean, 1Q, 3Q
        :return: nothing
        """
        description = self.numerical_description(feature)
        if inline is False:
            print('\'', feature.name, '\'', sep='')
            for k, v in description.items():
                print('  · {:<4s}: {:.04f}'.format(k, v))
            return
        else:
            body = ('{}({:<.4}) ' * len(description))[:-1]
            values = [(k, str(description[k])) for k in description]
            values_flattened = list(sum(values, ()))
            body_formatted = body.format(*values_flattened)
            return body_formatted

    def describe(self, feature_name=None, inline=False):
        """
        Wrapper.
        Calls the proper feature description method, depending on whether the
        feature is numerical or categorical. If no arguments are passed, the
        description of the entire dataset is provided.
        :param feature_name: the feature
        :param inline: whether the output is multiple lines or inline.
        :return: the string, only when inline=True
        """
        if feature_name is None:
            return self.describe_dataset()

        # It could happen that target has not yet been defined.
        target_name = None if self.target is None else self.target.name

        # If feature specified, ensure that it is contained somewhere
        assert feature_name in (list(self.features) + [target_name])

        if feature_name == target_name:
            feature = self.target
        else:
            feature = self.features[feature_name]
        if feature.dtype.name in self.categorical_dtypes:
            return self.describe_categorical(feature, inline)
        else:
            return self.describe_numerical(feature, inline)

    def summary(self, what='all'):
        """
        Printout a summary of each feature.
        :type what: the list of columns to be summarized: all, numerical,
        categorical, etc.
        :return: N/A
        """
        assert what in self.meta_tags

        max_width = 25
        max_len_in_list = np.max([len(s) for s in list(self.select(what))]) + 2
        if max_len_in_list > max_width:
            max_width = max_len_in_list
        else:
            max_width = max_len_in_list
        formatting = '{{:<{}s}}: {{:<10s}} {{}}'.format(max_width)
        print('Features Summary ({}):'.format(what))
        for feature_name in list(self.select(what)):
            feature_formatted = '\'' + feature_name + '\''
            print(formatting.format(
                feature_formatted, self.select(what)[feature_name].dtype.name,
                self.describe(feature_name, inline=True)))
        return

    def table(self, which='all', max_width=80):
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
        return

    @staticmethod
    def plot_correlation_matrix(corr_matrix):
        plt.subplots(figsize=(11, 9))
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=0.75, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5});
        plt.show();
        return

    def plot_double_density(self, feature, category=None):
        """
        Double density plot between a feature and a reference category.
        :param feature: The name of a feature in the dataset.
        :param category: The name of the reference category we want to
        represent the double density plot against. If None, then the target
        variable is used.
        :return: None

        Example:
            # represent multiple density plots, one per unique value of the
            # target
            my_data.double_density(my_feature)

            # represent double density plots, one per unique value of the
            # categorical feature 'my_feature2'
            my_data.double_density(my_feature1, my_categorical_feature2)
        """
        # Get the list of categories
        if category is None or self.target.name == category:
            categories = self.target.unique()
            category_series = self.target
        else:
            assert category in list(self.categorical), \
                '"category" must be a categorical feature'
            categories = self.features[category].unique()
            category_series = self.features[category]

        assert feature in self.numerical, '"Feature" must be numerical.'
        # plot a density for each value of the category
        for value in categories:
            sns.distplot(self.features[feature][category_series == value],
                         hist=False, kde=True,
                         kde_kws = {'shade': True},
                         label=str(value))
            # self.features[feature][category_series == value].plot(
            #     kind='density', label=str(value))
