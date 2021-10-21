import operator
import math
import argparse
import logging

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from sklearn.utils.validation import check_random_state

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x', ARG1='y')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Simple training model
samples = rng.uniform(-1, 1, 100).reshape(50, 2)
values = samples[:, 0]**2 - samples[:, 1]**2 + samples[:, 1] - 1

def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    diff = numpy.sum((func(*samples.T) - values)**2)
    return diff

toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main(POP, GEN, CXPB, PC, DATFILE, INSTANCES):
    
    ## population size
    pop = toolbox.population(n=POP)
    hof = tools.HallOfFame(1)

    ## Probabilities for Crossover, Mutation and number of generations (iterations)
    NGEN = GEN
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, CXPB, MUTPB, 40, stats=mstats, halloffame=hof, verbose=True)

    logging.debug("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    logging.debug("  Evaluated %i individuals" % len(pop))

    # Begin the evolution
    for g in range(NGEN):
        logging.debug("\n-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        #invalid_ind = [ind for ind in offspring]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            #logging.debug("Ind: {} - Fit: {}".format(ind, fit))

        logging.debug("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    logging.debug("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    logging.debug("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    predictions, probas, score = evaluate(best_ind, X_test, y_test)

    with open(DATFILE, 'w') as f:
        f.write(str(score))

    pool.close()

# Main function
if __name__ == "__main__":

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
    main(args.pop,args.gen, args.cros, args.pc, args.datfile, args.file)
    
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
