import itertools
import random

import numpy as np
from deap import base, creator, tools
from deap.algorithms import eaMuPlusLambda
from scipy.special import binom
from tqdm import tqdm

n_talks = 22
n_talks_per_slot = n_talks // 2
n_person = 180

talks = np.arange(n_talks)

PRIOS = []
for n in range(n_person):
    prio = list(range(5)) + list(np.repeat(100, n_talks - 5))
    np.random.shuffle(prio)
    PRIOS.append(prio)
PRIOS = np.array(PRIOS)


def assigned_priorities(prios, phenotype):
    assigned_prios = []
    for prio in prios:
        assigned_prio = [
            np.min(prio[phenotype[0]]),
            np.min(prio[phenotype[1]])
        ]
        assigned_prios.append(assigned_prio)

    return np.array(assigned_prios)


def compute_costs(prios, phenotype):
    return np.sum(assigned_priorities(prios, phenotype))


def to_phenotype_(genotype):
    return np.where(np.array(genotype) == 0), np.where(np.array(genotype) == 1)


def to_phenotype(genotype):
    return np.where(genotype == 0), np.where(genotype == 1)


def brute_force(prios):
    best = None, np.inf
    for slot1 in tqdm(itertools.combinations(talks, n_talks_per_slot),
                      total=binom(n_talks, n_talks_per_slot)):
        genotype = np.zeros(n_talks)
        genotype[list(slot1)] = 1

        phenotype = to_phenotype_(genotype)

        costs = compute_costs(prios, phenotype)

        if costs < best[1]:
            best = phenotype, costs

    return best


def ga(prios):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    def random_individual():
        ind = [1] * n_talks_per_slot + [0] * n_talks_per_slot
        random.shuffle(ind)
        return creator.Individual(ind)

    toolbox = base.Toolbox()
    toolbox.register("individual", random_individual)
    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual,
                     n=40)

    def evaluate(genotype):
        return compute_costs(prios, to_phenotype(genotype)),

    toolbox.register("evaluate", evaluate)

    def mate(genotype0, genotype1):
        return genotype0, genotype1

    toolbox.register("mate", mate)

    def mutate(genotype):
        genotype_ = genotype.copy()
        maxrand = len(genotype_) - 1
        idx1 = random.randint(0, maxrand)
        idx2 = random.randint(0, maxrand)
        genotype_[idx1] = genotype[idx2]
        genotype_[idx2] = genotype[idx1]
        return creator.Individual(genotype_),

    toolbox.register("mutate", mutate)

    toolbox.register("select", tools.selBest)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("min", np.min)

    hof = tools.HallOfFame(maxsize=1, similar=np.array_equal)

    return eaMuPlusLambda(population=toolbox.population(),
                          toolbox=toolbox,
                          mu=20,
                          lambda_=20,
                          cxpb=0.0,
                          mutpb=1.0,
                          ngen=100,
                          stats=stats,
                          halloffame=hof,
                          verbose=True), hof


(pop, logbook), hof = ga(PRIOS)
elitist = to_phenotype(hof[0])

print(
    f"Best individual: Slot 0 are talks {elitist[0]}, slot 1 are talks {elitist[1]}"
)
print(assigned_priorities(PRIOS, elitist))
