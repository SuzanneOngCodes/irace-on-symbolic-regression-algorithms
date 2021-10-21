import operator
import math
import random
import argparse
import logging

import numpy
import pandas as pd
import feyn

ql = feyn.connect_qlattice()

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
train, test = feyn.tools.split(X, ratio=(3,1), random_state=42)

def main(RAN,EPO,COMP,DATFILE):
    ql.reset(random_seed=RAN)
    
    # Setting semantic types
    stypes = {'color': 'c'}
    
    # Set number of epochs
    n_epochs = EPO
    
    # Initialize the list of models
    models = []
    # Sample and fit
    for epoch in range(n_epochs):
        
        # Sample models (no data here yet)
        models += ql.sample_models(
            input_names=train.columns,
            output_name='alcohol',
            kind='regression',
            stypes=stypes,
            max_complexity=COMP
        )
        
        # Fit the models with train data
        models = feyn.fit_models(models, train, loss_function='squared_error')
        
        # Remove redundant and worst performing models
        models = feyn.prune_models(models)
        
        # Display best model of each epoch
        feyn.show_model(models[0], label=f"Epoch: {epoch}", update_display=True)
        
        # Update QLattice with the models sorted by loss
        ql.update(models)
    
    best = models[0]

    with open(DATFILE, 'w') as f:
        f.write(str(best))

if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
        
    ap = argparse.ArgumentParser(description='Symbolic Regression with Feyn using QLattice')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--ran', dest='ran', type=int, required=True, help='Population size')
    ap.add_argument('--epo', dest='epo', type=float, required=True, help='Crossover probability')
    ap.add_argument('--comp', dest='comp', type=float, required=True, help='Mutation probability')
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
