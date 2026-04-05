import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import PandasDataset
import logging
import yaml
import platform
import sklearn
import joblib
from pathlib import Path

#Logging
logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger=logging.getLogger(__name__)

#Argument parser
def parse_args():
    parser=argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config",type=str,required=True,help="Path to model_config.yaml")
    parser.add_argument("--data",type=str,required=True,help="Path to processed CSV dataset")
    parser.add_argument("--models-dir",type=str,required=True,help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri",type=str,default=None,help="MLflow tracking URI")
    return parser.parse_args()

#Load model config
def get_model_instance(name,params):
    model_map={
        "LinearRegression":LinearRegression,
        "RandomForest":RandomForestRegressor,
        "GradientBoosting":GradientBoostingRegressor,
        "XGBoost":xgb.XGBRegressor
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

def main(args):
    #Load model config
    with open(args.config,"r") as f:
        config = yaml.safe_load(f)
    model_cfg=config["model"]

    #Set up MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg["name"])
    
    #Load data
    data=pd.read_csv(args.data)
    target=model_cfg["target_variable"]

    #Selecting features from training experimentation
    features_sets=model_cfg["feature_sets"]["rfe"]
    X=data[features_sets].astype('float64')
    y=data[target].astype('float64')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    #Get model
    model=get_model_instance(model_cfg["best_model"],model_cfg["parameters"])

    #Start MLflow
    with mlflow.start_run(run_name="final_training") as run:
        if not run:
            raise RuntimeError("CRITICAL: MLflow run failed to initialize. Aborting to prevent data loss")
        
        logger.info(f"Training model: {model_cfg['best_model']}")
        model.fit(X_train,y_train)
        y_pred=model.predict(X_test)
        mae=float(mean_absolute_error(y_test,y_pred))
        r2=float(r2_score(y_test,y_pred))


        #Log params and metrics
        mlflow.log_params(model_cfg['parameters'])
        mlflow.log_metrics({"mae":mae,"r2":r2})
        run_id=run.info.run_id
        #Creating signature
        signature=infer_signature(X_test,y_pred)
        
        #Log datasets
        dataset_df=X_train.copy()
        dataset_df[target]=y_train
        dataset:PandasDataset=mlflow.data.from_pandas(dataset_df,source=args.data,name="training_dataset")
        mlflow.log_input(dataset,context="training")


        #Log and register model
        mlflow_sklearn.log_model(sk_model=model,
                                 name="tuned_model",
                                 registered_model_name=model_cfg["name"],
                                 signature=signature,
                                 input_example=X_train.iloc[:5],
                                 pip_requirements=[
                                     f"scikit-learn=={sklearn.__version__}",
                                     f"xgboost=={xgb.__version__}",
                                     f"pandas=={pd.__version__}",
                                     f"numpy=={np.__version__}"
                                     ]         
                                    )


        model_name=model_cfg['name']
        # Set alias
        client=MlflowClient()
        #Getting last version
        versions=client.search_model_versions(filter_string=f"name= '{model_name}'")
        if versions:
            latest_version_num=max(int(v.version) for v in versions)
            client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=str(latest_version_num)
                )
            logger.info(f"Set 'champion' alis to version {latest_version_num}")
        else:
            logger.info(f"No versions found for model, alias not set")
        
        # Add a human_readable description
        description=(
            f"Model for predicting hpuse prices.\n"
            f"Algorithm:{model_cfg['best_model']}\n"
            f"Hyperparameters: {model_cfg['parameters']}\n"
            f"Features used: Only features that where selected after RFE process {model_cfg['feature_sets']['rfe']} \n"
            f"Target variable: {target}\n"
            f"Trained on dataset: {args.data}\n"
            f"Model saved at: {args.models_dir}/trained/{model_name}.pkl\n"
            f"Performance Metrics:\n"
            f" - MAE: {mae:.2f}\n"
            f" - R2: {r2:.4f}\n" 
        )
        client.update_registered_model(name=model_name,description=description)

        # Add tags for better organization

        client.set_registered_model_tag(name=model_name,key="algorithm",value=model_cfg["best_model"])
        client.set_registered_model_tag(name=model_name,key="hyperparameters",value=str(model_cfg["parameters"]))
        client.set_registered_model_tag(name=model_name,key="features",value=model_cfg["feature_sets"]["rfe"])
        client.set_registered_model_tag(name=model_name,key="target_variable",value=target)
        client.set_registered_model_tag(name=model_name,key="training_dataset",value=args.data)
        client.set_registered_model_tag(name=model_name,key="model_path",value=f"{args.models_dir}/trained/{model_name}.pkl")

        # Add dependency tags
        deps={
            "python_version":platform.python_version(),
            "scikit_learn_version":sklearn.__version__,
            "xgboost_version":xgb.__version__,
            "pandas_version":pd.__version__,
            "numpy_version":np.__version__
        }

        for k,v in deps.items():
            client.set_registered_model_tag(model_name,k,v)
        
        #Save model and features locally

        save_path=f"{args.models_dir}/trained/{model_name}.pkl"
        save_path_features=f"{args.models_dir}/trained/features.pkl"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model,save_path)
        joblib.dump(model_cfg["feature_sets"]["rfe"],save_path_features)
        logger.info(f"Saved trained model to: {save_path}")
        logger.info(f"Final MAE: {mae:.4f}, R2: {r2:4f}")

if __name__=="__main__":
    args=parse_args()
    main(args)