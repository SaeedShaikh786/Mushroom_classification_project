from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,Evaluate_models
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os
import sys
import pandas as pd
from dataclasses import dataclass



@dataclass
class ModelTrainerConfig:
    ''' Model trainer path object path stored '''
    model_trainer_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_path=ModelTrainerConfig()

    def initiate_model_training(self,train_arr,test_arr):
        ''' Stored the model.pkl into artifacts '''
        try:
            logging.info(f"{train_arr.shape,test_arr.shape}")
            logging.info("Model trainer stage:")
            X_train=train_arr[:,:-1]
            X_test=test_arr[:,:-1]
            y_train=train_arr[:,-1]
            y_test=test_arr[:,-1]

            logging.info("train and test data gathered")

            models={"RandomForest":RandomForestClassifier(),
            "LogisticRegression":LogisticRegression(),
            "SVC":SVC(probability=True),
            "DecisionTree":DecisionTreeClassifier(),
            "GradientBoost":GradientBoostingClassifier(),
            "AdaBoost":AdaBoostClassifier()}


            params={"RandomForest":{'criterion':['gini', 'entropy'],'max_features':['sqrt','log2'],"ccp_alpha":[0.02,0.03,0.04]},
            "LogisticRegression":{"penalty":["l1","l2"],"solver":["saga"],"C":[0.02,0.03]}
                
            ,"SVC":{"C":[0.02,0.03,0.05],"gamma":["scale","auto"]}
                
                ,"DecisionTree":{'criterion':['gini', 'entropy'],
            'splitter':['best','random'],"ccp_alpha":[0.02,0.03]}
                
                ,"GradientBoost":{'learning_rate':[
                    0.03,0.04],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],'n_estimators': [8,16,32,50,75]}
                
                ,"AdaBoost":{"n_estimators":[16,32,46,50],"learning_rate":[0.01,0.02,0.03]}}

            
            Report:dict=Evaluate_models(X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,models=models,params=params)
            #logging.info(pd.DataFrame(Report,index=False))
            max_val=max(list(Report.values()))
            
            index=list(Report.values()).index(max_val)
            best_model=list(models.values())[index];best_model
            logging.info(f"Best model name is :{list(models.keys())[index]} and Score is {max_val}")

            save_object(self.model_path.model_trainer_path,obj=best_model)

        except Exception as e:
            logging.error("Error Occured in Model Trainer")
            raise CustomException(error_message=e,error_detail=sys)