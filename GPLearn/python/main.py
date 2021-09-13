import argparse
import logging
import matplotlib.pyplot as plt
from multiprocess import Pool
import numpy as np
import pandas as pd
import random
import sys
import operator
import math
from sympy import sympify

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_equal, assert_almost_equal
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function

# Graphs and dataset
import matplotlib.pyplot as plt
import seaborn as sb
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

# Dictonary
converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
    'sin': lambda x    : sin(x),
    'cos': lambda x    : cos(x),
    'inv': lambda x: 1/x,
    'sqrt': lambda x: x**0.5,
    'pow3': lambda x: x**3
}

def main(POP, CXPB, MUTPB, DATFILE):

    function_set = ['add', 'sub', 'mul', 'div','cos','sin','neg','inv']
    
    # line =plt.plot(graph)
    # plt.show()

    # line =plt.plot(num_features)
    # plt.show()
    
    # To ensure CXPB + MUTPB are 1 or less than
    Prob = MUTPB + CXPB
    if Prob > 1:
        MUTPB = 1 - CXPB
    
    est_gp = SymbolicRegressor(population_size=POP,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=CXPB, p_subtree_mutation=MUTPB,
                           p_hoist_mutation=0, p_point_mutation=0,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
                           
    est_gp.fit(X_train, y_train)
    print('R2:',est_gp.score(X_test,y_test))
    next_e = sympify((est_gp._program), locals=converter)
    next_e

    with open(DATFILE, 'w') as f:
        f.write(str(est_gp.score(X_test,y_test)*100))


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
