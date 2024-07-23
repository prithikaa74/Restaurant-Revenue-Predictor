import sys
from dataclasses import dataclass
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocesssor.pkl')
    
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformation_object(self):
          
            try:
                categorical_cols=['Cuisine','Location','Parking Availability']
                numerical_cols=['Rating','Seating Capacity','Average Meal Price','Marketing Budget','Social Media Followers','Chef Experience Years','Number of Reviews','Avg Review Length','Ambience Score','Service Quality Score','Weekend Reservations','Weekday Reservations']
                
                cuisine_categories=['Japanese','Mexican','Italian','Indian','French','American']
                location_categories=['Rural','Downtown','Suburban']
                park_categories=['Yes','No']
                
                                
                num_pipeline=Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='median')),
                        ('scaler',StandardScaler())
                        ]
                )

                cat_pipeline=Pipeline(
                steps=[
                        ('imputer',SimpleImputer(strategy='most_frequent')),
                        ('ordinal_encoder',OrdinalEncoder(categories=[cuisine_categories,location_categories,park_categories])),
                        ('scaler',StandardScaler())
                        ]
                )

                preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
                    ]
                )
                
                return preprocessor
                
            except Exception as e:
                raise CustomException(e,sys)
            
    def initiate_data_transformation(self,train_path,test_path):
            try:
            
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)
                
                
                preprocessor_obj=self.get_data_transformation_object()
                target_column='Revenue'
                drop_column=target_column
                print(train_df)
                input_column_train_df=train_df.drop(columns=['Revenue'],axis=1)
                target_column_train_df=train_df[target_column]
                
                input_column_test_df=test_df.drop(columns=drop_column,axis=1)
                target_column_test_df=test_df[target_column]
                
                input_column_train_arr=preprocessor_obj.fit_transform(input_column_train_df)
                input_column_test_arr=preprocessor_obj.transform(input_column_test_df)

                save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                            obj=preprocessor_obj
                            )      
                
                return(
                    input_column_train_arr,
                    input_column_test_arr,
                    target_column_train_df,
                    target_column_test_df,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
                
            except Exception as e:
                raise CustomException(e,sys)