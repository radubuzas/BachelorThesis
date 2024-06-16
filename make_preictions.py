import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import category_encoders as ce

from typing import Literal

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

sampling_seed: int = 1970
seed: int = 1233123

####################### DATA PROCESSING #######################

def get_df() -> pd.DataFrame:
    print('Awaiting connection...')
    cnx = sqlite3.connect('anon_jobs.db3')

    print('Connected to database!\n\n')

    print('Reading data from database...')
    df = pd.read_sql_query("SELECT * FROM 'Jobs';", cnx)
    print('Done reading data from database!\n\n')


    df = df.sort_values(by='SubmitTime', ascending=True)

    cnx.close()
    
    return df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    constant_columns = [column for column in df.columns if df[column].nunique() == 1]

    df_processed = df.drop(constant_columns, axis=1)

    df_processed = df_processed.drop(['JobID'], axis=1)
    df_processed = df_processed.drop(['SubmitTime'], axis=1)
    df_processed = df_processed.drop(['RunTime'], axis=1)
    df_processed = df_processed.drop(['WaitTime'], axis=1)

    return df_processed

def get_processed_df() -> pd.DataFrame:
    df = get_df()
    df = preprocess_df(df)

    return df

####################### END OF DATA PROCESSING #######################


####################### DATA ENCODING #######################
def encode(X_train, Y_train, X_test, type: str = ('leave_one_out', 'target'), **kwargs):
    categorical_columns = X_train.select_dtypes(include='object').columns

    if type == 'leave_one_out':
        encoder = ce.LeaveOneOutEncoder(cols=categorical_columns, return_df=True, **kwargs)
    elif type == 'target':
        encoder = ce.TargetEncoder(cols=categorical_columns, return_df=True)
    else:
        raise ValueError('Invalid type')
    
    X_train = encoder.fit_transform(X_train, Y_train)
    X_test = encoder.transform(X_test)

    return X_train, X_test, encoder


####################### END OF DATA ENCODING #######################

####################### MODELS #######################

def print_store_scores(model, X_test, Y_test) -> None:
    Y_pred = model.predict(X_test)

    print(f'Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}')
    print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(Y_test, Y_pred))}')
    print(f'Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred)}')
    print(f'Mean Absolute Percentage Error: {mean_absolute_percentage_error(Y_test, Y_pred)}')
    print(f'R^2: {model.score(X_test, Y_test)}')


def linear_regression_model(X_train, Y_train, X_test, Y_test):
    model = LinearRegression()

    model.fit(X_train, Y_train)

    print_store_scores(model, X_test, Y_test)

    return model

def knn_model(X_train, Y_train, X_test, Y_test, **kwargs) -> KNeighborsRegressor:
    model = KNeighborsRegressor(n_jobs=-1, **kwargs)
    if not kwargs:
        param_grid = {  
            'n_neighbors': np.arange(1, 200),
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }

        grid_search = GridSearchCV(estimator=model,
                                    param_grid=param_grid,
                                    scoring='neg_mean_absolute_percentage_error',
                                    cv=5,
                                    verbose=10,
                                    n_jobs=-1)
        
        grid_search.fit(X_train, Y_train)
        model = KNeighborsRegressor(n_jobs=-1, **grid_search.best_params_)
        print(f'Best parameters: {grid_search.best_params_}')
    
    model.fit(X_train, Y_train)
    
    print_store_scores(model, X_test, Y_test)

    return model

def regression_tree_model(X_train, Y_train, X_test, Y_test, **kwargs) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(random_state=seed, **kwargs)

    if not kwargs:
        param_grid = {
                'max_depth': np.arange(1, 40),
                'min_samples_split': np.arange(2, 10),
                'min_samples_leaf': np.arange(1, 10)
        }

        grid_search = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                scoring='neg_root_mean_squared_error',
                                cv=5,
                                verbose=10,
                                n_jobs=-1)
        
        grid_search.fit(X_train, Y_train)
        model = DecisionTreeRegressor(random_state=seed, **grid_search.best_params_)
        print(f'Best parameters: {grid_search.best_params_}')
    
    model.fit(X_train, Y_train)
    
    print_store_scores(model, X_test, Y_test)

    return model


def random_forest_model(X_train, Y_train, X_test, Y_test, **kwargs) -> RandomForestRegressor:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    model = RandomForestRegressor(random_state=seed, n_jobs=-1, **kwargs)

    if not kwargs:

        param_distributions = {
            'n_estimators': randint(1, 100),
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 11),
            'min_samples_leaf': randint(1, 20),
        }

        random_search = RandomizedSearchCV(estimator=model,
                                            param_distributions=param_distributions,
                                            n_iter=100,  # Number of parameter settings sampled
                                            cv=5,
                                            scoring='neg_root_mean_squared_error',
                                            random_state=seed,
                                            n_jobs=-1,
                                            verbose=10)


        random_search.fit(X_train, Y_train)
        model = RandomForestRegressor(random_state=seed, n_jobs=-1, **random_search.best_params_)
        print(f'Best parameters: {random_search.best_params_}')

    model.fit(X_train, Y_train)
    
    print_store_scores(model, X_test, Y_test)

    return model



def xgboost_model(X_train, Y_train, X_test, Y_test, **kwargs) -> XGBRegressor:
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    model = XGBRegressor(random_state=seed, n_jobs=-1, **kwargs)

    if not kwargs:
        param_distributions = {
            'n_estimators': randint(100, 1000),
            'max_depth': randint(10, 50),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.4, 0.6),
            'colsample_bytree': uniform(0.4, 0.6),
            'colsample_bylevel': uniform(0.4, 0.6),
            'colsample_bynode': uniform(0.4, 0.6),
            'gamma': uniform(0, 10),
            'reg_alpha': [0, 1, 5],
            'reg_lambda': [0, 1, 5],
        }

        random_search = RandomizedSearchCV(estimator=model,
                                            param_distributions=param_distributions,
                                            n_iter=100,  # Number of parameter settings sampled
                                            cv=5,
                                            scoring='neg_root_mean_squared_error',
                                            random_state=seed,
                                            n_jobs=-1,
                                            verbose=10)

        random_search.fit(X_train, Y_train)
        model  = XGBRegressor(random_state=seed, n_jobs=-1, **random_search.best_params_)
        print(f'Best parameters: {random_search.best_params_}')

    model.fit(X_train, Y_train)
    
    print_store_scores(model, X_test, Y_test)

    return model

####################### END OF MODELS #######################

if __name__ == '__main__':
    df = get_processed_df()

    # N = int(input('Enter number of processes: '))
    args = {}

    args_model = {}
    # args_model = {'n_neighbors': 7, 'p': 1, 'weights': 'distance'}
    # args_model = {'n_neighbors': 3, 'p': 1}
    # args_model = {'max_depth': 16, 'min_samples_leaf': 1, 'min_samples_split': 3}
    # args_model = {'max_depth': 16, 'min_samples_leaf': 3, 'min_samples_split': 9}

    N = 30_000
    fraction = 0.1
    encoding = 'leave_one_out'
    if encoding == 'leave_one_out':
        args = {'sigma': 0.5, 'random_state': seed}

    model_name: Literal['linear', 'knn', 'rt', 'rf', 'xgb'] = 'knn'

    df_sample = df.sample(n=N, weights=df.groupby('UsedCPUTime')['UsedCPUTime'].transform('sum'), random_state=sampling_seed)

    target = 'UsedCPUTime'

    X = df_sample.drop(target, axis=1)
    Y = df_sample[target]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    X_train, X_test, encoder = encode(X_train, Y_train, X_test, type=encoding, **args)

    X_total = encoder.transform(df.drop([target], axis=1))
    Y_total = df[target]

    print(f'Training model {model_name}...')

    if model_name == 'linear':
        model = linear_regression_model(X_train, Y_train, X_test, Y_test)
    elif model_name == 'knn':
        model = knn_model(X_train, Y_train, X_test, Y_test, **args_model)
    elif model_name == 'rt':
        model = regression_tree_model(X_train, Y_train, X_test, Y_test, **args_model)
    elif model_name == 'rf':
        model = random_forest_model(X_train, Y_train, X_test, Y_test, **args_model)
    elif model_name == 'xgb':
        model = xgboost_model(X_train, Y_train, X_test, Y_test, **args_model)
    else:
        raise ValueError('Invalid model name')

    print(score:= model.score(X_total, Y_total))

    # save predictions
    if score > 0.8:
        file_name = f'predictions_{model_name}_{encoding}_{np.round(score, 3)}.npy'
        print(f'Saving predictions to: {file_name}')
        np.save(file_name, model.predict(X_total))

        import pickle

        with open(f'{model_name}_{encoding}_s_{np.round(score, 3)}.pkl', 'wb') as f:
            pickle.dump(model, f)
