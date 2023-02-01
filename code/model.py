import pandas as pd
import numpy as np
from sklearn.metrics import classification_report as score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rd_clasifier
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

path = 'database/'
data_frame = pd.read_csv(path + "desafio_manutencao_preditiva_treino.csv")
target_variable = 'failure_type'

# data cleansing and transformation
check_null_values = data_frame.isnull().values.any()
print('Is there any null value? ' + str(check_null_values))

data_cleansed = data_frame.drop(columns=['udi', 'product_id', 'type'])

def data_preparation(data_cleansed, target_variable):
    df = data_cleansed.dropna()
    df['target_name_encoded'] = df[target_variable].replace(({
        'No Failure': 0, 'Power Failure': 1, 'Tool Wear Failure': 2, 'Overstrain Failure': 3,
        'Random Failures': 4, 'Heat Dissipation Failure': 5
    }))

    X = df.drop(columns=[target_variable, 'target_name_encoded'])
    y = df['target_name_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    return X, y, X_train, X_test, y_train, y_test

X, y, X_train, X_test, y_train, y_test = data_preparation(data_cleansed, target_variable)

# models classification
xgb_clf_gbtree = XGBClassifier(booster='gbtree', 
                               objective='multi:softmax', 
                               num_class=5)
xgb_clf_gbtree.fit(X_train, y_train.ravel(), )
y_pred_gb = xgb_clf_gbtree.predict(X_test)
results_log_xgb_gbtree = score(y_test, y_pred_gb)
print('Classification report XGB with gbtree:' + ".\n" + results_log_xgb_gbtree)

xgb_clf_gblinear = XGBClassifier(booster='gblinear', 
                        objective='multi:softmax',
                        num_class=5)
xgb_clf_gblinear.fit(X_train, y_train)
y_pred_li = xgb_clf_gblinear.predict(X_test)
results_log_xgb_gblinear = score(y_test, y_pred_li)
print('Classification report XGB with gblinear:' + ".\n" + results_log_xgb_gblinear)

random_for_clf = rd_clasifier(max_depth=2, random_state=0)
random_for_clf.fit(X_train, y_train)
y_pred_fo = random_for_clf.predict(X_test)
results_log_ran_for = score(y_test, y_pred_fo)
print('Classification report Random Forest:' + ".\n" + results_log_ran_for)

# run selected model
df_test = pd.read_csv(path + "desafio_manutencao_preditiva_teste.csv")
df_test_cleansed = df_test.drop(columns=['udi', 'product_id', 'type'])
test_predictions = xgb_clf_gbtree.predict(df_test_cleansed)

final_result = pd.DataFrame({'rowNumber': df_test.index,
                             'predictedValues': test_predictions})
final_result.to_csv('predicted.csv', index= False)
