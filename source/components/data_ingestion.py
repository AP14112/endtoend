import os 
import sys
from source.exception import customException
from source.logger import logging
import pandas as pd
from source.components.data_transformation import dataTransformation
from source.components.data_transformation import datatransformationConfig
from source.components.data_transformation import datatransformationConfig
from source.components.model_training import model_trainer_config
from source.components.model_training import ModelTrainer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')


class dataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method or component")
        try:
            df=pd.read_csv('notebooks\data\StudentsPerformance.csv')
            logging.info("read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split initated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion completed successfully")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise customException(e,sys)
if __name__=="__main__":
    obj=dataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()    
    data_transformation=dataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))