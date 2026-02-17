from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def clean_df(df, ndata_column_label='Data Points', ndata_limit=3):
    '''
    Remove rows with too few data points.
    '''
    try:
        good_flag = df[ndata_column_label] > ndata_limit
    except:
        print(f'Unable to find {ndata_column_label} column in input df. Exiting.')
        sys.exit()
    clean_df = df[good_flag]
    return clean_df


def get_world_data(df, world, world_column_label='World'):
    '''
    Isolate all rows for a specific world.
    '''
    world = str(world)
    try:
        data_world = df[df[world_column_label] == world]
    except:
        print(f'Unable to find {world_column_label} column OR that world has no data in the input df. Exiting.')
        sys.exit()
    return data_world


def get_xy_traintest(df_world, test_size_, feature_list):
    '''
    Create the train/test feature and target sets for classification.
    '''
    X = df_world[feature_list]

    # convert continuous average spawn time (0–45 min) into 3 categories: early, mid, late
    try:
        y_cont = df_world['Average']
    except:
        print(f'Unable to find "Average" column in input df. Exiting.')
        sys.exit()

    bins = [0, 15, 30, 45]
    labels = ['early', 'mid', 'late']
    y = pd.cut(y_cont, bins=bins, labels=labels, include_lowest=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_, random_state=60, stratify=y)
    return X_train, X_test, y_train, y_test


def create_model(X_train, X_test, y_train, y_test):
    '''
    Create and test a Random Forest Classifier.
    '''
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # performance metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {acc:.3f}")
    print(f"Weighted F1-score: {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_test, y_pred


def plot_results(y_test, y_pred, world):
    '''
    Plot a confusion matrix to visualize model performance.
    '''
    cm = confusion_matrix(y_test, y_pred, labels=['early', 'mid', 'late'])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.title(f'Confusion Matrix – World {world}')
    plt.colorbar(label='Count')
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ['early', 'mid', 'late'])
    plt.yticks(tick_marks, ['early', 'mid', 'late'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # overlay numbers
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.tight_layout()
    plt.show()


def run_all(df, clean=True, ndata_limit=3, world='301', feature_list=None, test_size_=0.3):
    '''
    Full pipeline to clean, isolate, train, and visualize classification performance.
    '''
    if feature_list is None:
        feature_list = ['Week Number']   
        print('Defaulting to "Week Number" only for the feature list...')

    #if True, remove points with too few data points
    if clean:
        df = clean_df(df, ndata_limit=ndata_limit)

    #isolate all rows for a specific world
    data_world = get_world_data(df, world)

    #create training/testing data
    xtrain, xtest, ytrain, ytest = get_xy_traintest(data_world, feature_list=feature_list, test_size_=test_size_)

    #create model and predict
    ytest, ypred = create_model(xtrain, xtest, ytrain, ytest)

    #visualize results
    plot_results(ytest, ypred, world)


if __name__ == '__main__':
    print('EXAMPLE:\n'
          'run_all(df, clean=True, ndata_limit=3, world="301",'
          'feature_list=["Week Number"], test_size_=0.3)')