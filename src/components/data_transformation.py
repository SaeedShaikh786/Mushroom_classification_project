import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.utils import save_object

@dataclass()
class DataTransformationConfig:
        preprocessor_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.get_preprocessor_path=DataTransformationConfig()

    def get_preprocessor_obj(self):
        try:
            preprocessor=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
                    # use sparse_output=False --or it will give error in np.c_[,]
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, error_detail=sys)
    
    def InitiateDataTransformation(self,train_path,test_path):
        ''' Return preprocessor path and transformed array'''
        try:
            logging.info("Data initiation has started")
            train_data=pd.read_csv(train_path)

            test_data=pd.read_csv(test_path)

            x_train,y_train=train_data.drop("class",axis=1),train_data[["class"]]
            x_test,y_test=test_data.drop("class",axis=1),test_data[["class"]]
            logging.info("splitting the data has done")

            preprocessor=self.get_preprocessor_obj()

            x_train_arr=preprocessor.fit_transform(x_train)
            
            
            x_test_arr=preprocessor.transform(x_test)
            logging.info(f"x_train and X_test array shapes are {x_train_arr.shape},{x_test_arr.shape}")
            logging.info(f"y_train and y_test array shapes are {y_train.shape},{y_test.shape}")

            train_array=np.c_[x_train_arr,np.array(y_train)]
            #train_array = np.concatenate((x_train_arr, np.array(y_train).reshape(-1, 1)), axis=1)
            logging.info(pd.DataFrame(train_array))

            logging.info(f"train_arr {train_array.shape}")

            test_array=np.c_[x_test_arr,np.array(y_test)]
            logging.info(f"test_arr {test_array.shape}")

            logging.info(f"y_train and y_test array shapes are {y_train.shape},{y_test.shape}")

            save_object(self.get_preprocessor_path.preprocessor_path,preprocessor)

            logging.info("Preprocessor object saved")
            return (self.get_preprocessor_path.preprocessor_path,
            train_array,test_array)


        except Exception as e:
            logging.info(f"error occured in Data-transformation:{e}")
            raise CustomException(e,error_detail=sys)

