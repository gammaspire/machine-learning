'''
AIM: run Random Forest Classification after performing k-means clustering with different values of k (2-10). I want to output the 

'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, f1_score, 
                             classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split

import sys
sys.path.insert(0,'../utils')

from clustering_utils import run1_kmeans
from feature_utils import get_feature_names
from init_parameters import Params
params = Params()

def run_rfc_comparison(df, k_list, n_init=10):
    
    feature_labels = get_feature_names(params)
    
    if type(k_list) is not list:
        print('k must be a list of integers.')
        return
    
    for k in k_list:
        #run kmeans function; outputs df with Feature Cluster labels
        feature_data = run1_kmeans(df, feature_labels, k=k, n_init=n_init, random_state=42)

        #define the actual X and y sets for RFC
        X = feature_data[feature_labels]
        y = feature_data['Feature Cluster']

        #define the training and test sets
        #I use 60% of the sample for training and 40% for testing
        #random_state ensures reproducibility. can set to any positive integer.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

        #model scaffold
        model = RandomForestClassifier(n_estimators=50, max_depth=14,class_weight='balanced',
                                       random_state=42,max_features='sqrt')
        #train the model...
        model.fit(X_train, y_train)

        #predict Feature Groups (classes) of test dataset
        y_pred = model.predict(X_test)
        
        print('#'*20)
        print()
        print(f'k = {k}')
        print(classification_report(y_test, y_pred, digits=3))
        print()
        print(confusion_matrix(y_test, y_pred, labels=list(range(k))))
        print()
        print('#'*20)