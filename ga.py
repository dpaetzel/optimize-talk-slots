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
persons = np.arange(n_person)

PRIOS = []
for n in range(n_person):
    prio = list(range(5)) + list(np.repeat(100, n_talks - 5))
    np.random.shuffle(prio)
    PRIOS.append(prio)
PRIOS = np.array(PRIOS)

TALKERS = []
talkers_ = random.sample(list(persons),n_talks)
TALKERS = dict(zip(talkers_,talks))
# update prios of talkers
for talker, talk in TALKERS.items():
    PRIOS[talker][talk] = 0

def assigned_priorities(prios, phenotype):
    assigned_prios = []
    for prio in prios:
        assigned_prio = [
            np.min(prio[phenotype[0]]),
            np.min(prio[phenotype[1]])
        ]
        assigned_prios.append(assigned_prio)

    return np.array(assigned_prios)


def punish_erroneously_assigned_talkers(talkers, phenotype):
    cost = [] # cost for talks where talker is not present in talks
    for talker, talk in talkers.items():
        c = 0
        #punish if their first prio is not in the same slot as their talk
        if talk in phenotype[0][0]:
            if 0 in phenotype[1][0]:
                c += 1000
            #talkers should get a prio that is not in the same slot as their talk --> punish if this isnt the case
            if 1 in phenotype[0][0] or 2 in phenotype[0][0]: 
                c += 10
        else:
            if 0 in phenotype[0][0]:
                c += 1000
            if 1 in phenotype[1][0] or 2 in phenotype[0][0]:
                c += 10
        cost.append(c)
    return np.array(cost)


def compute_costs(prios, talkers, phenotype):
    return np.sum(assigned_priorities(prios, phenotype)) + np.sum(punish_erroneously_assigned_talkers(talkers, phenotype))


def to_phenotype_(genotype):
    return np.where(np.array(genotype) == 0), np.where(np.array(genotype) == 1)


def to_phenotype(genotype):
    return np.where(genotype == 0), np.where(genotype == 1)


def brute_force(prios, talkers):
    best = None, np.inf
    for slot1 in tqdm(itertools.combinations(talks, n_talks_per_slot),
                      total=binom(n_talks, n_talks_per_slot)):
        genotype = np.zeros(n_talks)
        genotype[list(slot1)] = 1

        phenotype = to_phenotype_(genotype)

        costs = compute_costs(prios, talkers, phenotype)

        if costs < best[1]:
            best = phenotype, costs

    return best


def ga(prios, talkers):
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
        return compute_costs(prios, talkers, to_phenotype(genotype)),

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


costs = 0
individual, costs = brute_force(PRIOS, TALKERS)
print(f"Best BruteFroce result: {individual} with costs: {costs}")

(pop, logbook), hof = ga(PRIOS, TALKERS)
elitist = to_phenotype(hof[0])

print(
    f"Best individual: Slot 0 are talks {elitist[0]}, slot 1 are talks {elitist[1]}"
)
print(assigned_priorities(PRIOS, elitist))
print(f"Costs GA: {logbook[-1]['min']} vs BruteForce: {costs}")

