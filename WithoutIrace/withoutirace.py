import argparse
import logging
import numpy as np
import pandas as pd
import random
import sys
import operator
from math import sqrt
import math

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
import graphviz
from collections import OrderedDict
from sympy import simplify

# For Support Vector Regression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold


# For Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

# For GPLearn
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# For FFX estimator
import ffx

# For QLattice
import feyn
# ql = feyn.connect_qlattice()

# For DEAP
from tpot import TPOTRegressor

from pyshgp.push.instruction_set import InstructionSet
from pyshgp.gp.estimators import PushEstimator
from pyshgp.gp.genome import GeneSpawner

# Checks time
from time import process_time

# Import writer class from csv module
from csv import writer

# GPLearn
from gplearn.genetic import SymbolicRegressor
# SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Symbolic regression parameters
pop = [int(x) for x in np.linspace(start = 50, stop = 500, num = 400)]
generation = [int(x) for x in np.linspace(start = 10, stop = 50, num = 40)]
crossover = [float(x) for x in np.linspace(0.01, 1, num = 99)]
parsimony_coefficient = [float(x) for x in np.linspace(start = 0.01, stop = 1, num=99)]
mut = [int(x) for x in np.linspace(start = 0, stop = 0, num = 99)]
cv = [int(x) for x in np.linspace(start = 2, stop = 20, num = 20)]

# Random forest parameters
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 500, num = 500)]
n_jobs = [int(x) for x in np.linspace(start = 1, stop = 20, num = 15)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 20, num = 15)]
verbose = [int(x) for x in np.linspace(start = 0, stop = 20, num = 20)]

# SVR parameters
c = [int(x) for x in np.linspace(start = 1, stop = 20, num = 15)]
epsilon = [float(x) for x in np.linspace(start = 0.1, stop = 20, num = 200)]
cache_size = [float(x) for x in np.linspace(0.001,1.0, num = 90)]
max_iter = [int(x) for x in np.linspace(start = 1, stop = 100, num = 99)]

arr = {'../Datasets/vladislavleva-1.training.csv',
        '../Datasets/vladislavleva-2.training.csv',
        '../Datasets/vladislavleva-3.training.csv',
        '../Datasets/vladislavleva-4.training.csv',
        '../Datasets/vladislavleva-5.training.csv',
        '../Datasets/vladislavleva-6.training.csv',
        '../Datasets/vladislavleva-7.training.csv',
        '../Datasets/vladislavleva-8.training.csv',
        '../Datasets/keijzer-1.training.csv',
        '../Datasets/keijzer-2.training.csv',
        '../Datasets/keijzer-3.training.csv',
        '../Datasets/keijzer-4.training.csv',
        '../Datasets/keijzer-5.training.csv',
        '../Datasets/keijzer-6.training.csv',
        '../Datasets/keijzer-7.training.csv',
        '../Datasets/keijzer-8.training.csv',
        '../Datasets/keijzer-9.training.csv',
        '../Datasets/keijzer-10.training.csv',
        '../Datasets/keijzer-11.training.csv',
        '../Datasets/keijzer-13.training.csv',
        '../Datasets/keijzer-14.training.csv',
        '../Datasets/keijzer-15.training.csv',
        '../Datasets/nguyen-3.training.csv',
        '../Datasets/nguyen-4.training.csv',
        '../Datasets/nguyen-5.training.csv',
        '../Datasets/nguyen-6.training.csv',
        '../Datasets/nguyen-7.training.csv',
        '../Datasets/nguyen-8.training.csv',
        '../Datasets/nguyen-9.training.csv',
        '../Datasets/nguyen-10.training.csv'}
        
def RandomizedCV(regressor):
    
    random.seed(12345)
        
    # Dataset
    for i in arr:
        start = process_time()
        df = pd.read_csv(i)
            
        size = len(df.columns)-1
        X = df.drop(['target'], axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state=1)
            
        name = ""
        global rf_random
        
        if regressor == "GPLearn":
            name = "GP"
            rf = SymbolicRegressor(p_subtree_mutation=0,
                           p_hoist_mutation=0, p_point_mutation=0,
                           max_samples=0.9, verbose=1,random_state=0)
            random_grid = {'population_size': pop,
                       'generations': generation,
                   'p_crossover': crossover,
                   'parsimony_coefficient': parsimony_coefficient}
            rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid)
        elif regressor == "RR":
            name = "RR"
            rf = RandomForestRegressor()
            random_grid = {'n_estimators': n_estimators,
                       'n_jobs': n_jobs,
                       'max_depth': max_depth,
                       'verbose': verbose}
            rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid)
    #    elif regressor == "TPOT":
    #        name = "TPOT"
    #        rf = TPOTRegressor()
    #        random_grid = {'population_size': pop,
    #                   'generations': generation,
    #                   'crossover_rate': crossover,
    #                   'mutation_rate':mut,
    #                   'cv': cv}
    #        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid)
        else:
            name = "SVM"
            random_grid = {'rbf_svm__C': c,
                       'rbf_svm__epsilon': epsilon,
                       'rbf_svm__cache_size': cache_size,
                       'rbf_svm__max_iter': max_iter}
            steps = [('scaler', MinMaxScaler()), ('rbf_svm', SVR())]
            pipeline = Pipeline(steps)
            # do search
            rf_random = RandomizedSearchCV(pipeline, param_distributions =random_grid)
        # Fit the random search model
        rf_random.fit(X_train, y_train)

        scores = rf_random.score(X_test, y_test)
        
        # Fit training sets into model
        y_test_pred = rf_random.predict(X_test)
        
        acc = float(np.sum(y_test == y_test_pred)) / y_test.shape[0]
        print("Error: %.6f" %(1-acc))
        
        # Mean absolute error
        mae = mean_absolute_error(y_test, y_test_pred)
        # Mean squared error to check performance
        mse = mean_squared_error(y_test, y_test_pred)
        # Root mean squared error to check accuracy
        rmse = sqrt(mse)
        print("Data 1: "+str(rmse) + " , ")
        print(rf_random.best_params_)
        print("\n")
        
        stop = process_time()
        tt = stop-start
        
        # Record CPU time, dataset, mae, mse and rmse into csv
        List=[tt, mae, mse , rmse , i, name, scores, rf_random.best_score_, rf_random.best_params_]
        with open('time.csv', 'a') as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)
            # Pass the list as an argument into the writerow()
            writer_object.writerow(List)
            # Close the file object
            f_object.close()


    
# Main function
if __name__ == "__main__":
    RandomizedCV("GPLearn")
    RandomizedCV("RR")
    RandomizedCV("SVR")

