import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from source.exception import customException
from source.logger import logging
from source.utils import save_object
from source.utils import evaluate_models

@dataclass
class model_trainer_config:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=model_trainer_config()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("self training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models={
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression":LinearRegression(),
            "K-Neighbours classifer":KNeighborsRegressor(),
            "XGBClassifier":XGBRegressor(),
            "CatBossting Classifier":CatBoostRegressor(verbose=False),
            "Adaboost Classifier": AdaBoostRegressor(),
            }
        
            Model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models)
            best_model_score=max(sorted(Model_report.values()))
            best_model_name=list(Model_report.keys())[
                list(Model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise customException("No best model found")
            logging.info("best model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted) 
            return r2_square    
        except Exception as e:
            raise customException(e,sys)


