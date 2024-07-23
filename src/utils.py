import pandas as pd
import numpy as np
from src.exception import CustomException
import sys
import os
import pickle
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)


def model_evaluate(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for model_name,model in models.items():
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            r_square=r2_score(y_test,y_pred)
            mse=mean_squared_error(y_test,y_pred)*100
            report[model_name]=r_square,mse
        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)