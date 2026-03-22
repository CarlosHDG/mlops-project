import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
from datetime import datetime
import joblib


logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger=logging.getLogger("feature-engineering")

def create_features(df:pd.DataFrame):
    df_featured=df.copy()
    logger.info(f"Creating features")
    # House age
    df_featured["house_age"]=datetime.now().year-df_featured["year_built"]
    logger.info(f"House age feature created")
    # Price per sqft
    # df_featured["price_per_sqft"]=df_featured["price"]/df_featured["sqft"]
    # df_featured["price_per_sqft"]=df_featured["price_per_sqft"].replace([np.inf,-np.inf],np.nan).fillna(0)
    # logger.info(f"Price per sqft feature created")
    # bed_bathroom_ratio
    df_featured["bed_bath_ratio"]=df_featured["bedrooms"]/df_featured["bathrooms"]
    df_featured["bed_bath_ratio"]=df_featured["bed_bath_ratio"].replace([np.inf,-np.inf],np.nan).fillna(0)
    logger.info(f"Bed bath ratio feature created")
    return df_featured

def create_preprocessor(df:pd.DataFrame)->ColumnTransformer:
    logger.info(f"Creating preprocessing pipeline.")
    #Drop target value and year_built feature (house_age instead of year_built)
    df=df.drop(columns=["price","year_built"],errors="ignore")
    #Defining feature groups
    numerical_features=df.select_dtypes(include="number").columns.to_list()
    categorical_features=df.select_dtypes(exclude="number").columns.to_list()
    #Preprocessing for numerical features
    numerical_transformer=Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="mean"))
        ])
    #Preprocessing for categorical features
    categorical_transformer=Pipeline(steps=[
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ])
    preprocessor=ColumnTransformer(
        transformers=[
            ("num",numerical_transformer,numerical_features),
            ("cat",categorical_transformer,categorical_features)
        ]
    )
    return preprocessor
def run_feature_engineering(input_file:str,output_file:str,preprocessor_output_path:str):
    logger.info(f"Loading cleaned data from {input_file}")
    df=pd.read_csv(input_file)
    #Create features
    df_featured=create_features(df)
    logger.info(f"Features created, dataset with shape {df_featured.shape}")

    #Preprocessing
    preprocessor=create_preprocessor(df_featured)
    X=df_featured.drop(columns=["price"],errors="ignore")
    y=df_featured["price"] if "price" in df_featured.columns else None
    X_transformed=preprocessor.fit_transform(X)
    logger.info(f"Fit and transformed applied")
    colums_name=preprocessor.get_feature_names_out()
    logger.info(f"New columns features are \n {colums_name}")
    #Save preprocessor
    joblib.dump(preprocessor,preprocessor_output_path)
    logger.info(f"Saved preprocessor to {preprocessor_output_path}")

    #Saved fully processed data
    df_transformed=pd.DataFrame(X_transformed,columns=colums_name)
    if y is not None:
        df_transformed["price"]=y.values
    df_transformed.to_csv(output_file,index=False)
    logger.info(f"Data processed saved to {output_file}")
    return df_transformed

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument("--input",required=True,help="Cleaned data (CSV) path")
    parser.add_argument("--output",required=True,help="Featured data outputh path")
    parser.add_argument("--preprocessor",required=True,help="Path for saving the preprocessor")
    args=parser.parse_args()
    run_feature_engineering(args.input,args.output,args.preprocessor)