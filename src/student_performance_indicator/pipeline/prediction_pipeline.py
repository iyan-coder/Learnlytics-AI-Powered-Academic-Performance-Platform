import sys

import pandas as pd

from student_performance_indicator.exception.exception import (
    StudentPerformanceException,
)
from student_performance_indicator.utils.main_utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame) -> pd.Series:
        try:
            model_path = "final_model/model.pkl"
            preprocessor_path = "final_model/preprocessor.pkl"

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            transformed_data = preprocessor.transform(features)
            prediction = model.predict(transformed_data)

            return prediction

        except Exception as e:
            raise StudentPerformanceException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(data_dict)

        except Exception as e:
            raise StudentPerformanceException(e, sys)
