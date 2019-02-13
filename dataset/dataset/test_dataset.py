import pandas as pd
import warnings
warnings.simplefilter(action='ignore')

from dataset import Dataset
from unittest import TestCase


class TestDataset(TestCase):
    def setUp(self):
        self.df1 = pd.DataFrame(
            data={'col1': [1, 2, 3], 'col2': ['a', 'a', 'b']})
        self.ds = Dataset.from_dataframe(self.df1)

    def test_select(self):
        self.assertEqual(list(self.ds.select('numerical')), ['col1'])
