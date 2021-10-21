import argparse
import logging
import numpy as np
import pandas as pd
import sys
import operator
import math
import random
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# For Support Vector Regression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Checks time
from time import process_time

# Import writer class from csv module
from csv import writer

# GPLearn
from gplearn.genetic import SymbolicRegressor

def main(C, EPI, CS, ITER, DATFILE, INSTANCES):
    random.seed(0)

    # Dataset
    df = pd.read_csv(INSTANCES)
    size = len(df.columns)-1
    X = df.drop(['target'], axis=1)
    y = df['target']

    # dividing X and y into training and testing units
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    est_svr = make_pipeline(StandardScaler(), SVR(C=C, epsilon=EPI, cache_size = CS, max_iter = ITER))
                           
    # Fit training sets into model
    est_svr.fit(X_train, y_train)
    y_test_pred = est_svr.predict(X_test)
    # Mean absolute error
    mae = mean_absolute_error(y_test, y_test_pred)
    # Mean squared error to check performance
    mse = mean_squared_error(y_test, y_test_pred)
    # Root mean squared error to check accuracy
    rmse = sqrt(mse)
    print(rmse)
    
    with open(DATFILE, 'w') as f:
        f.write(str(rmse))
        
    return mae,mse,rmse
        
# Main function
if __name__ == "__main__":
    start = process_time()
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
        
    ap = argparse.ArgumentParser(description='Symbol Vector Regression')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--c', dest='c', type=float, required=True, help='Strength of the regularization')
    ap.add_argument('--epi', dest='epi', type=float, required=True, help='Epsilon')
    ap.add_argument('--cs', dest='cs', type=float, required=True, help='Size of the kernel cache (in MB)')
    ap.add_argument('--iter', dest='iter', type=int, required=True, help='Max iterations')
    ap.add_argument("-f", "--file", type=str, required=True)
    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug(args)
    mae,mse,rmse = main(args.c,args.epi,args.cs,args.iter,args.datfile, args.file)
    stop = process_time()
    tt = stop-start
    args.file = args.file.replace(str(pathlib.Path().resolve()) + "/../Datasets/", "")
    
    # Record CPU time, dataset, mae, mse and rmse into csv
    List=[tt, mae, mse , rmse , args.file, args.c,args.epi,args.cs,args.iter]
    with open('time.csv', 'a') as f_object:
  
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
      
        # Pass the list as an argument into the writerow()
        writer_object.writerow(List)
      
        # Close the file object
        f_object.close()
