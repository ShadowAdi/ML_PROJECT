import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


from src.exception import CustomException
from src.logger import logging


from src.utils import save_object,evaluateModel


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artificats","model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.modelTrainerConfig=ModelTrainerConfig()

    def initiateModelTrainer(self,train_array,test_array):
        try:
            logging.info("Split Train And Test input data")
            X_train,y_train,X_test,y_test=(
                                           train_array[:,:-1],
                                           train_array[:,-1],
                                           test_array[:,:-1],
                                           test_array[:,-1]
                                           )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting Regressor":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbors Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "Cat Boosting Regressor":CatBoostRegressor(),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            model_report:dict=evaluateModel(X_train,y_train,X_test,y_test,models)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best Model Score Found")


            save_object(file_path=self.modelTrainerConfig.trained_model_file_path,
                        obj=best_model)
            

            predicted=best_model.predict(X_test)

            r2_score_ans=r2_score(y_test,predicted)
            return r2_score_ans

        except CustomException as e:
            CustomException(e,sys)
