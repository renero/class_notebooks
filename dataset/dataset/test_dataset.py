import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import pandas as pd
from dataset import Dataset
from unittest import TestCase

warnings.simplefilter(action='ignore')


class TestDataset(TestCase):

    df1 = pd.DataFrame(
        data={'col1': [1, 2, 3, 2, 2, 2, 1, 3, 2, 1],
              'col2': ['a', 'a', 'b', 'a', 'b', 'a', 'a', 'a', 'b', 'c'],
              'col3': ['1', '1', '1', '0', '0', '1', '1', '0', '1', '0']
              })

    def setUp(self):
        self.ds = Dataset.from_dataframe(self.df1)

    def test_numbers_to_float(self):
        self.assertIs(self.ds.features.col1.dtype, np.dtype('float64'))

    def test_select(self):
        self.assertEqual(list(self.ds.select('numerical')), ['col1'])

    def test_set_target(self):
        self.ds.set_target('col3')
        self.assertNotIn('col3', self.ds.features)
        self.assertEqual(self.ds.target.name, 'col3')

    def test_update(self):
        self.ds.describe()
        self.ds.summary()
        self.assertEqual(self.ds.meta['features'], list(self.df1))
        self.assertEqual(self.ds.meta['target'], 'col3')
        self.assertIs(self.ds.numerical, ['col1'])
        self.assertIs(self.ds.data, self.ds.features)
        self.assertEqual(self.ds.categorical, 'col2')
