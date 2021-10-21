import argparse
import logging
import numpy as np
import pandas as pd
import random
import sys
import operator
import math
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Checks time
from time import process_time

# Import writer class from csv module
from csv import writer

def main(NUM, DEPT, JOB, VER, DATFILE, INSTANCES):
    random.seed(0)
    
    # Dataset
    df = pd.read_csv(INSTANCES)
    size = len(df.columns)-1
    X = df.drop(['target'], axis=1)
    y = df['target']

    # dividing X and y into training and testing units
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=1)
    
    est_rf = RandomForestRegressor(n_estimators=NUM, max_depth=DEPT,n_jobs= JOB, verbose=VER, random_state=0)
        # Fit training sets into model
    est_rf.fit(X_train, y_train)
    y_test_pred = est_rf.predict(X_test)
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

if __name__ == "__main__":
    start = process_time()
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
        
    ap = argparse.ArgumentParser(description='Random Forest Regressor')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--num', dest='num', type=int, required=True, help='Number of estimators (trees)')
    ap.add_argument('--dep', dest='dep', type=int, required=True, help='Max depth')
    ap.add_argument('--jobs', dest='jobs', type=int, required=True, help='Number of jobs to run in parallel')
    ap.add_argument('--ver', dest='ver', type=int, required=True, help='Verbosity')
    ap.add_argument("-f", "--file", type=str, required=True)
    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
    
    args = ap.parse_args()

    mae,mse,rmse = main(args.num,args.dep,args.jobs,args.ver,args.datfile, args.file)
    stop = process_time()
    tt = stop-start
    args.file = args.file.replace(str(pathlib.Path().resolve()) + "/../Datasets/", "")
    
    # Record CPU time, dataset, mae, mse and rmse into csv
    List=[tt, mae, mse , rmse , args.file,args.num,args.dep,args.jobs,args.ver]
    with open('time.csv', 'a') as f_object:
  
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
      
        # Pass the list as an argument into the writerow()
        writer_object.writerow(List)
      
        # Close the file object
        f_object.close()
