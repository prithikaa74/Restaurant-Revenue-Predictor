import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.utils import save_object
from src.utils import model_evaluate
from src.exception import CustomException

from dataclasses import dataclass
import sys
import os

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_data_input,test_data_input,train_data_target,test_data_target):
        try:
            X_train,y_train,X_test,y_test=train_data_input,train_data_target,test_data_input,test_data_target
            models={
            'LinearRegression':LinearRegression(),
            'Support vector Machine':SVR(),
            #'DTR':DecisionTreeRegressor(),
            'RandomForest':RandomForestRegressor(),
            'Neighbors':KNeighborsRegressor(),
            'Gradient Boosting': GradientBoostingRegressor()
            
            }

            model_report:dict=model_evaluate(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n==================================')
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(f'Best Model Found,Model Name :{best_model_name},accuracy:{best_model_score}')
            print('\n===============================================')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e,sys)