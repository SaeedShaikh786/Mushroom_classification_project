
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import  DataTransformation
from src.components.model_trainer import ModelTrainer 
from src.components.data_ingestion import DataIngestion



if __name__=="__main__":
    try:
        logging.info("data ingestion has started")

        obj=DataIngestion()
        train_path,test_path=obj.Initiate_data_ingestion()
        logging.info("Data Transformaton has started")
        trans_obj=DataTransformation()
        _,train_array,test_array=trans_obj.InitiateDataTransformation(train_path,test_path)
        logging.info("Model Trainer has initiated")
    # always check for the return sequences
        model_trainer=ModelTrainer()
        
        model_trainer.initiate_model_training(train_arr=train_array,test_arr=test_array)
    except Exception as e:
        raise CustomException(e,sys)



