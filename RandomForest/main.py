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
from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

# Checks time
from time import process_time

# Import writer class from csv module
from csv import writer

def main(NUM, DEPT, JOB, VER, ID, DATFILE, INSTANCES, SEED):
    random.seed(12345)
    ID = ID.replace(str(pathlib.Path().resolve()) + "/Instances/" , "")
    
    # Dataset
    df = pd.read_csv(INSTANCES)
        
    size = len(df.columns)-1
    X = df.drop(['target'], axis=1)
    y = df['target']
    
    # dividing X and y into training and testing units
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state=1)
    n_train = math.ceil(X.shape[0] * 0.7)

    X_train = X.iloc[:n_train, :]
    y_train = y.iloc[:n_train]

    X_test = X.iloc[n_train:, :]
    y_test = y.iloc[n_train:]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    # if instance starts with "i", use i-th cv-split on trainings data
    if ID.startswith("i"):
        fold_indx = int(ID[1:]) - 1
        print("Use %dth-cv split" %(fold_indx))
        kfold = KFold(n_splits=10, shuffle=True, random_state=12345)
        kf = list(kfold.split(X=X_train))
        train_index, test_index = kf[fold_indx]
        X_train_fold = X_train.iloc[train_index, :]
        y_train_fold = y_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index, :]
        y_test_fold = y_train.iloc[test_index]
        X_train, y_train = X_train_fold, y_train_fold
        X_test, y_test = X_test_fold, y_test_fold
    
    est_rf = RandomForestRegressor(n_estimators=NUM, max_depth=DEPT,n_jobs= JOB, verbose=VER, random_state=SEED)
        # Fit training sets into model
    est_rf.fit(X_train, y_train)
    y_test_pred = est_rf.predict(X_test)
    
    acc = float(np.sum(y_test == y_test_pred)) / y_test.shape[0]
    print("Error: %.6f" %(1-acc))
    
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
    ap.add_argument('--ins', dest='ins', type=str, required=True, help='Instance')
    ap.add_argument('--i', dest='i', type=str, required=True, help='Instance')
    ap.add_argument('--s', dest='s', type=int, required=True, help='Seed')
    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')
    
    args = ap.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        
    logging.debug(args)

    mae,mse,rmse = main(args.num,args.dep,args.jobs,args.ver,args.ins,args.datfile, args.i,args.s)
    stop = process_time()
    tt = stop-start
    args.i = args.i.replace(str(pathlib.Path().resolve()) + "/../Datasets/", "")
    
    # Record CPU time, dataset, mae, mse and rmse into csv
    List=[tt, mae, mse , rmse , args.i,args.num,args.dep,args.jobs,args.ver]
    with open('time.csv', 'a') as f_object:
  
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
      
        # Pass the list as an argument into the writerow()
        writer_object.writerow(List)
      
        # Close the file object
        f_object.close()
