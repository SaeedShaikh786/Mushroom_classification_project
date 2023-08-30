import pickle
import os
from src.exception import CustomException
import sys
from src.logger import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score,roc_auc_score

def Evaluate_models(X_train,X_test,y_train,y_test,models,params):
    ''' Return Model report '''

    try:
        report={}
        for i in range(len(models)):
        
            model=list(models.values())[i]
            param=list(params.values())[i]
    
            Random_cv=RandomizedSearchCV(estimator=model,param_distributions=param,cv=3)
            Random_cv.fit(X_train,y_train)
    
            model.set_params(**Random_cv.best_params_)
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            #y_score=model.predict_proba(X_test)[:,1]
            #roc_score=roc_auc_score(y_test,y_score)

            score=f1_score(y_test,y_pred,average="macro")
            logging.info(f"Scores from utils:model->{list(models.keys())[i]}:{score}")
            report[list(models.keys())[i]]=score
            #logging.info("Report from utils",report)
        return report
    except Exception as e:
        raise CustomException(e,error_detail=sys)




def save_object(file_path,obj):
    """ 
    This function is used to save the object..object may be preprocessing or model.pkl
    """
    try:
        dir_name=os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,"wb") as file:

           pickle.dump(obj,file)
    except Exception as e:
        raise CustomException(e, error_detail=sys)



def load_obj(file_path):
    '''
    To load the object that are saved using the save_obj function
    '''
    try:

        with open(file_path,"rb") as file_obj:
        
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)