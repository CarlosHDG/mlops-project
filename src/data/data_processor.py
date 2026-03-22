import pandas as pd
import numpy as np
from pathlib import Path
import logging
#Logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger=logging.getLogger("data-processor")

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df:pd.DataFrame):
    logger.info(f"Cleaning data")
    logger.info("Cleaning dataset")
    df_cleaned=df.copy()
    # Handling missing values
    # For target values
    missing_target_values=df_cleaned["price"].isnull().sum()

    if missing_target_values>0:
        logger.info(f"Found {missing_target_values} in target value")
        df_cleaned=df_cleaned.dropna(subset=["price"])
        logger.info(f"Dropped {missing_target_values} rows")
    
    for column in df_cleaned.columns:
        missing_values=df_cleaned[column].isnull().sum()
        if missing_values>0:
            logger.info(f"Found {missing_values} in column :{column}")
            # For numerical columns
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value=df_cleaned[column].median()
                df_cleaned[column]=df_cleaned[column].fillna(median_value)
                logger.info(f"Filled {missing_values} missing values in {column} with median: {median_value}")
            # For categorical columns
            else:
                mode_value=df_cleaned[column].mode()[0]
                df_cleaned[column]=df_cleaned[column].fillna(mode_value)
                logger.info(f"Filled {missing_values} missing values in {column} with mode: {mode_value}")

    #Handling outliers in target variable using IQR
    logger.info(f"Handling outliers with IQR")
    Q1=df_cleaned["price"].quantile(0.25)
    Q3=df_cleaned["price"].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR

    count_outliers=((df["price"]>upper_bound) | (df["price"]<lower_bound)).sum()
    if count_outliers>0:
        logger.info(f"Found {count_outliers} in target value (price)")
        df_cleaned=df_cleaned[(df["price"]<=upper_bound) & (df["price"]>=lower_bound)]
        logger.info(f"Removed {count_outliers} outliers. New dataset shape: {df_cleaned.shape}")
    else:
        logger.info(f"There aren't outliers in target value (price)")
    return df_cleaned

def process_data(input_file:str,output_file:str):
    output_path=Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    df=load_data(input_file)
    df_cleaned=clean_data(df)
    df_cleaned.to_csv(output_file,index=False)
    logger.info(f"Cleaned dataset saved in {output_file}")
    return df_cleaned

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Data processing raw data")
    parser.add_argument("--input",default="data/raw/house_data.csv",help="Input raw data path")
    parser.add_argument("--output",default="data/processed/cleaned_house_data.csv",help="Output cleaned data path")

    args=parser.parse_args()
    process_data(input_file=args.input,output_file=args.output)