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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

# Checks time
from time import process_time

# Import writer class from csv module
from csv import writer

# GPLearn
from gplearn.genetic import SymbolicRegressor

def main(POP, GEN, CXPB, PC, DATFILE, INSTANCES):
    random.seed(0)

    # Dataset
    df = pd.read_csv(INSTANCES)
    size = len(df.columns)-1
    X = df.drop(['target'], axis=1)
    y = df['target']

    # dividing X and y into training and testing units
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.5, test_size = 0.5, random_state=1)

    est_gp = SymbolicRegressor(population_size=POP,
                           generations=GEN, stopping_criteria=0.01,
                           p_crossover=CXPB, p_subtree_mutation=0,
                           p_hoist_mutation=0, p_point_mutation=0,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=PC, random_state=0)
                           
    # Fit training sets into model
    est_gp.fit(X_train, y_train)
    y_test_pred = est_gp.predict(X_test)
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
        
    ap = argparse.ArgumentParser(description='Genetic Programming with GPLearn')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
    ap.add_argument('--gen', dest='gen', type=int, required=True, help='Generations')
    ap.add_argument('--cros', dest='cros', type=float, required=True, help='Crossover probability')
    ap.add_argument('--pc', dest='pc', type=float, required=True, help='Parsimony coefficient')
    ap.add_argument("-f", "--file", type=str, required=True)
    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug(args)
    mae,mse,rmse = main(args.pop,args.gen, args.cros, args.pc, args.datfile, args.file)
    stop = process_time()
    tt = stop-start
    args.file = args.file.replace(str(pathlib.Path().resolve()) + "/../Datasets/", "")

    # Record CPU time, dataset, mae, mse and rmse into csv
    List=[tt, mae, mse , rmse , args.file,args.pop,args.gen, args.cros, args.pc]
    with open('time.csv', 'a') as f_object:
  
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
      
        # Pass the list as an argument into the writerow()
        writer_object.writerow(List)
      
        # Close the file object
        f_object.close()
