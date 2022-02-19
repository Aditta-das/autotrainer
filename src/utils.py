import os
from tabulate import tabulate
from IPython.display import display
from tqdm.auto import tqdm
import numpy as np
from logger import logger  
from sklearn.model_selection import train_test_split
from .cfg import CFG

def auto_output_folder(filename):
    base_path = os.path.dirname(os.getcwd())
    directory = os.path.join(base_path, filename)
    if not os.path.exists(directory):
        logger.info(f"Create folder name : {filename} folder")
        os.mkdir(directory)
    else:
        logger.info("Folder already exists")



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def null_checker(data):
	for col in data.columns:
		if data[col].isnull().sum() > 0:
			display(f"null col: {col} >> total null : {data[col].isnull().sum()}")
		

def label_encode():
	pass

def categorical_data():
	pass

def normal_data_split(df, label):
    X = df.drop(label, axis=1)
    y = df[label]
    xtrain, xtest, ytrain, ytest = train_test_split(
        X, y, random_state=CFG.random_state, shuffle=CFG.shuffle, test_size=CFG.test_size
    )
    return xtrain, xtest, ytrain, ytest
