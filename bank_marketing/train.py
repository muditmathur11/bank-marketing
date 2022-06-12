# Load  Pkgs
import pandas as pd
import random # delete later
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# Load Utils 
import dagshub # ml experiment tracking
# import mlflow # ml experiment tracking
import joblib
import os


# Load Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df

# Process Data
def split_data(df,label_col='label',test_size=0.3,output_path='data/processed/'):
    """Split Dataset into Features and Labels"""

    # Features & Labels
    Xfeatures = df.drop(label_col,axis=1)
    # Select last column of dataframe as a dataframe object
    # last_column = df.iloc[: , -1:]
    ylabels = df[label_col]
    # Split Dataset
    x_train,x_test,y_train,y_test = train_test_split(Xfeatures,ylabels,test_size=test_size,random_state=7)
    print("Generating Dataset for {}".format('training'))
    x_train.to_csv(os.path.join(output_path,"x_train.csv"))
    y_train.to_csv(os.path.join(output_path,"y_train.csv"))
    print("Generating Dataset for {}".format('testing'))
    x_test.to_csv(os.path.join(output_path,"x_test.csv"))
    y_test.to_csv(os.path.join(output_path,"y_test.csv"))
    print("Generating Metadata about dataset")
    col_xfeatures = pd.DataFrame(list(Xfeatures.columns))
    col_xfeatures.to_csv("features.csv",header=False,index=False)
    col_ylabels = pd.DataFrame({'target_labels':df['y'].unique()})
    col_ylabels.to_csv("target.csv",header=False,index=False)
    return x_train,x_test,y_train,y_test


def build_pipeline(Estimator,X,y,Transformer):
	"""Build a Pipeline using an Estimator and A Transformer

	>>> build_pipeline(LogisticRegression(),X,y,StandardScaler())
	"""
	ml_pipe = Pipeline(steps=[('scaler',Transformer),('clf',Estimator)])
	# Fit To Train
	ml_pipe.fit(X,y)
	return ml_pipe

def build_model(Estimator,X,y):
	"""Build a Model using an Estimator and Train on Dataset"""
	model = Estimator
	# Fit To Train
	model.fit(X,y)
	return model

def evaluate_model(ml_pipe,x_test,y_test):
    accuracy = ml_pipe.score(x_test,y_test)
    y_pred = ml_pipe.predict(x_test)
    f1score = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test,y_pred)
    return {'model_name':ml_pipe.named_steps['clf'],'accuracy':accuracy,'f1_score':f1score,'precision':precision}



# Usage
df = load_data("data/bank-additional.csv")
print("Preprocessing Data")
x_train,x_test,y_train,y_test = split_data(df,label_col='y')
# Build Models
pipe = build_pipeline(LogisticRegression(),x_train,y_train,StandardScaler())
print("Evaluating Model")
# Evaluate Model
evaluate_model(pipe,x_test,y_test)




with dagshub.dagshub_logger() as logger:
	print('Saving Model/Pipeline...')
	joblib.dump(pipe, 'models/pipe_lr.pkl')
	logger.log_hyperparams(model_class=type(pipe.named_steps['clf']).__name__)
	logger.log_hyperparams({'pipe_lr': pipe.get_params()})
	print("Working on Metrics...")
	# Log Metrics: This creates a metrics.csv file
	train_metrics = evaluate_model(pipe,x_test,y_test)
	print('Train metrics:')
	print(train_metrics)
	logger.log_metrics({f'train__{k}': v for k,v in train_metrics.items()})