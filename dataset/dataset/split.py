class Split:
    """
    This class represents a split from a dataset, it will assign
    each dataframe partition passed as argument to a different 
    attribute of the class: 'train', 'test' (and 'validation').
    The class method 'split' performs the splitting of the dataframe
    passed, according to the parameters passed.
    
    Example:
    
        from src import split
        
        X, Y = split.Split(my_dataframe, my_target_column)
        
    """
    split_name = ['train', 'test', 'validation']
    
    def __init__(self, splits):
        for index, partition in enumerate(splits):
            setattr(self, self.split_name[index], partition)