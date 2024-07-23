import sys
import os
from src.exception import CustomException
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocesssor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            print(f"Preprocessor path: {preprocessor_path}")
            print(f"Model path: {model_path}")

            # Check if the preprocessor and model files exist
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"No such file: {preprocessor_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No such file: {model_path}")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            if not isinstance(features, pd.DataFrame):
                raise ValueError("Features should be a pandas DataFrame")

            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                 Location: str,
                 Cuisine: str,
                 Rating: int,
                 Seating_Capacity: str,
                 price: str,
                 marketing: int,
                 followers: int,
                 experience: int,
                 reviews: int,
                 review_length: int,
                 ambience_score: int,
                 service_score: int,
                 parking: bool,
                 weekend_reservation: int,
                 weekday_reservation: int):
        
        self.Location = Location
        self.Cuisine = Cuisine
        self.Rating = Rating
        self.Seating_Capacity = Seating_Capacity
        self.price = price
        self.marketing = marketing
        self.followers = followers
        self.experience = experience
        self.reviews = reviews
        self.review_length = review_length
        self.ambience_score = ambience_score
        self.service_score = service_score
        self.parking = parking
        self.weekend_reservation = weekend_reservation
        self.weekday_reservation = weekday_reservation
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Location': [self.Location],
                'Cuisine': [self.Cuisine],
                'Rating': [self.Rating],
                'Seating Capacity': [self.Seating_Capacity],
                'Average Meal Price': [self.price],
                'Marketing Budget': [self.marketing],
                'Social Media Followers': [self.followers],
                'Chef Experience Years': [self.experience],
                'Number of Reviews': [self.reviews],
                'Avg Review Length': [self.review_length],
                'Ambience Score': [self.ambience_score],
                'Service Quality Score': [self.service_score],
                'Parking Availability': [self.parking],
                'Weekend Reservations': [self.weekend_reservation],
                'Weekday Reservations': [self.weekday_reservation]
            }

            df = pd.DataFrame(custom_data_input_dict)
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
