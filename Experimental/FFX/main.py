import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
import operator
import math

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import csr_matrix

# FFX estimator
import ffx

# Graphs
import matplotlib.pyplot as plt
import seaborn as sb

# Potential dataset
from pmlb import fetch_data

nsample = 400
sig = 0.2
x = np.linspace(-50, 50, nsample)
X = np.column_stack((x/5, 10*np.sin(x), (x-5)**3, np.ones(nsample)))
beta = [0.01, 1, 0.001, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)
df = pd.DataFrame()
df['x']=x
df['y']=y

X = df[['x']]
y = df['y']
y_true = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
def main(POP, CXPB, MUTPB, DATFILE):
    random.seed(320)
    
    # This creates a dataset of 2 predictors
#    X = np.random.random((POP, 2))
#    y = target(X)
#    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    FFX = ffx.FFXRegressor()
    FFX.fit(X_train, y_train)

    # line =plt.plot(num_features)
    # plt.show()
    

    with open(DATFILE, 'w') as f:
        f.write(str(FFX.score(X_test, y_test)*100))


if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
        
    ap = argparse.ArgumentParser(description='Feature Selection using GP with DEAP')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--pop', dest='pop', type=int, required=True, help='Population size')
    ap.add_argument('--cros', dest='cros', type=float, required=True, help='Crossover probability')
    ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability')
    ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug(args)

    main(args.pop, args.cros, args.mut, args.datfile)

#import pygraphviz as pgv
#
## [...] Execution of code that produce a tree expression
#
#nodes, edges, labels = graph(expr)
#
#g = pgv.AGraph()
#g.add_nodes_from(nodes)
#g.add_edges_from(edges)
#g.layout(prog="dot")
#
#for i in nodes:
#    n = g.get_node(i)
#    n.attr["label"] = labels[i]
#
#g.draw("tree.pdf")
