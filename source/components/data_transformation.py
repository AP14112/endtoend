import sys
from dataclasses import dataclass
import numpy as np
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from source.exception import customException
from source.logger import logging
from source.utils import save_object
@dataclass
class datatransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
class dataTransformation:
    def __init__(self):
        self.data_transformation_config=datatransformationConfig()
    def get_data_tranformer_obj(self):
        try:
            numerical_features=["writing score","reading score"]
            categorical_features=[
                'gender', 
                'race/ethnicity',
                'parental level of education', 
                'lunch', 
                'test preparation course'
            ]
            numerical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scalar",StandardScaler())
                ]
            )
            logging.info("numerical columns ecodded.....")
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scalar",StandardScaler(with_mean=False))
                ]
            )
            logging.info("categorical columns ecoddded...")
            preprocessor=ColumnTransformer(
                [
                    ("nume_pipeline",numerical_pipeline,numerical_features),
                    ("cat_pipeline",categorical_pipeline,categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise customException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train and test data completed")
            logging.info("obtaining preprocessing object")
            preprocessing_obj=self.get_data_tranformer_obj()
            target_column_name="math score"
            numerical_columns=["writing score","reading score"]
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            logging.info(
                f"Applying preprocessing on training and testing dataframe"
            )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(test_df.drop(columns=[target_column_name], axis=1))
            target_feature_test_df = test_df[target_column_name]
            train_arr=np.c_[
            input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
            input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info(f"saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise customException(e,sys)
