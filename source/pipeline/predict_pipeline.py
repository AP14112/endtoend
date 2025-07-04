import sys
import pandas as pd
from source.exception import customException
from source.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preproccesor_path='artifacts\preprocessor.pkl'
            model=load_object(model_path)
            preprocessor=load_object(preproccesor_path)
            data_scaled=preprocessor.transform(features)
            prediction=model.predict(data_scaled)
            return prediction
        except Exception as e:
            raise customException(e,sys)
class customData:
    def __init__( self,
        gender:str,
        ethnicity: str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int,
    ):
        self.gender=gender
        self.ethnicity=ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race/ethnicity":[self.ethnicity],
                "parental level of education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test preparation course":[self.test_preparation_course],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score],

            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise customException(e,sys)


