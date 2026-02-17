from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def clean_df(df, ndata_column_label='Data Points', ndata_limit=3):
    try:
        good_flag = df[ndata_column_label]>ndata_limit
    except:
        print(f'Unable to find {ndata_column_label} column in input df. Exiting.')
        sys.exit()
    clean_df = df[good_flag]
    return clean_df


def get_world_data(df, world, world_column_label='World'):
    world=str(world)
    #isolate all world data...
    try:
        data_world = df[df[world_column_label]==world]
    except:
        print(f'Unable to find {world_column_label} column OR that world has no data in the input df. Exiting.')
        sys.exit()
    return data_world


def get_xy_traintest(df_world, test_size_, feature_list):
    
    #the gaggle of features
    X = df_world[feature_list]

    #the 'target' quantity, log(M200)
    try:
        y = df_world['Average']
    except:
        print(f'Unable to find "Average" column in input df. Exiting.')
        sys.exit()
        
    #creating the train/test sets! I choose a random state and a test size of 0.2 (20% of sample)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_, random_state=60)
    
    return X_train, X_test, y_train, y_test


def create_model(X_train, X_test, y_train, y_test):

    #create the model!
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)  

    #train the model with the testing data defined above
    model.fit(X_train, y_train)

    #predict the log(M200) values with this newly-trained model using the test data
    y_pred = model.predict(X_test)

    #performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.3f}")
    print(f"R²: {r2:.3f} (|R|: {np.sqrt(np.abs(r2)):.3f})")

    return y_test, y_pred


def plot_results(y_test, y_pred, world):

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred,c=y_test-y_pred)

    #flag points with a difference within 1-sigma variance (+/- 7.5 minutes)
    var_flag = np.abs(y_test-y_pred) <= 7.5
    plt.scatter(y_test[var_flag], y_pred[var_flag], s=200, color='red', facecolor='None',
               label=r'Within 1$\sigma$ variance ($\pm 7.5$ min)')

    plt.xlabel('Actual Spawn Time [minutes]')
    plt.ylabel('Predicted Spawn Time [minutes]')
    plt.colorbar(label='Actual - Predicted Spawn Time')
    plt.legend()
    plt.title(f'World: {world}',fontsize=15)
    plt.show()


def run_all(df, clean=True, ndata_limit=3, world='301', feature_list=['Week Number'], test_size_=0.3):
    
    #if True, remove points with fewer than ndata_limit points
    if clean:
        df = clean_df(df, ndata_limit=ndata_limit)
    
    #isolate all rows for a specific world
    data_world = get_world_data(df, world)
    
    #quasi-randomly split x and y data; create training and testing sets
    xtrain, xtest, ytrain, ytest = get_xy_traintest(data_world, feature_list=feature_list, test_size_=test_size_)
    
    #generate model and test it
    ytest, ypred = create_model(xtrain, xtest, ytrain, ytest)
    
    #plot the results
    plot_results(ytest, ypred, world)
    

if __name__ == '__main__':
    
    print('EXAMPLE:\n'
          'run_all(df, clean=True, ndata_limit=3, world="301", feature_list=["Week Number"], test_size_=0.3)')
    