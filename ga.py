import codecs
from collections import defaultdict
import collections
import itertools
from pickle import FALSE
import pprint
import random
import sys

import numpy as np
from deap import base, creator, tools
from deap.algorithms import eaMuPlusLambda
from scipy.special import binom
from tqdm import tqdm
import survey_parsing

SYNTHETIC_DATA = False

if SYNTHETIC_DATA:
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

    # random init of TALKERS
    speakers_ = random.sample(list(persons), n_talks)
    speakers_ = dict(zip(speakers_, talks))
    TALKERS = speakers_
else:
    parsed_survey = survey_parsing.preprocess_excel(
        'Workshop Preference Voting Retreat 2022.xlsx')
    PRIOS = parsed_survey['prio_matrix']
    TALKERS = parsed_survey['speaker_workshop_dict']
    workshop_names = parsed_survey['workshop_list']
    participant_names = parsed_survey['id_to_name_dict']
    talks = set(TALKERS.values())
    n_talks = len(talks)

for speaker, talk in TALKERS.items():
    PRIOS[speaker][talk] = -500
n_talks_per_slot = n_talks // 2
workshops = set(TALKERS.values())


def assigned_priorities(prios, phenotype):
    assigned_prios = []
    for prio in prios:
        assigned_prio = [
            np.min(prio[phenotype[0]]),
            np.min(prio[phenotype[1]])
        ]
        assigned_prios.append(assigned_prio)

    return np.array(assigned_prios)


def workshop_attendance(assigned_prios, prios):
    workshops = defaultdict(list)
    for person, prios_for_person in enumerate(assigned_prios):
        for prio in prios_for_person:
            workshops[np.where(prios[person] == prio)[0][0]].append(person)
    return workshops  # WS->Attendees


def prettify_workshop_assignments(workshop_assignments, workshop_names, participant_names):
    workshops_pretty = dict()
    for workshop, participants in workshop_assignments.items():
        participants_pretty = [participant_names[participant]
                               for participant in participants]
        workshops_pretty[workshop_names[workshop]] = participants_pretty
    return workshops_pretty


def compute_costs(prios, talkers, phenotype):
    return np.sum(assigned_priorities(prios, phenotype))


def to_phenotype_(genotype):
    return np.where(np.array(genotype) == 0)[0], np.where(np.array(genotype) == 1)[0]


def to_phenotype(genotype):
    return np.where(genotype == 0)[0], np.where(genotype == 1)[0]


def brute_force(prios, talkers):
    best = None, np.inf
    bests = []
    for slot1 in tqdm(itertools.combinations(talks, n_talks_per_slot),
                      total=binom(n_talks, n_talks_per_slot)):
        genotype = np.zeros(n_talks)
        genotype[list(slot1)] = 1

        phenotype = to_phenotype(genotype)

        costs = compute_costs(prios, talkers, phenotype)

        if costs < best[1]:
            best = phenotype, costs
            bests = [best]
        if costs == best[1]:
            bests.append((phenotype, costs))

    return best, bests


def print_results(result, costs, prios, workshop_names, participant_names):
    pp = pprint.PrettyPrinter(width=140)
    print(
        f"\n------------------------------------------------------------------------------------------------------------------------------------------\n"
    )
    print(f'Workshop Assignment with Costs: {costs}')
    for i in range(2):
        print(f'\nWorkshops in Slot {i}\n')
        pp.pprint([workshop_names[ws] for ws in result[i]])

    print(f'\nParticipants per Workshop:\n')
    assigned_prios = assigned_priorities(prios, result)
    # ensure that every speaker is assigned to his workshop
    assert (assigned_prios == -500).sum(), len(TALKERS.items())
    if not SYNTHETIC_DATA:
        w_assignments = prettify_workshop_assignments(
            workshop_attendance(assigned_prios, prios), workshop_names, participant_names)
        pp.pprint(w_assignments)
        w_num = {workshop: len(participants)
                 for workshop, participants in w_assignments.items()}
        print(f'\nNumber of Participants per Workshop:\n')
        pp.pprint(w_num)


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


costs_bf = 0
best, bests = brute_force(PRIOS, TALKERS)
print('-----------------BRUTE FORCE-------------------')
stdout = sys.stdout
with codecs.open("results_bf.txt", "w", "utf-8-sig") as sys.stdout:
    for results_bf, costs_bf in bests:
        print_results(results_bf, costs_bf, PRIOS,
                      workshop_names, participant_names)
sys.stdout = stdout

(pop, logbook), hof = ga(PRIOS, TALKERS)
elitist = to_phenotype(hof[0])

print('-----------------GA-------------------')
costs_ga = logbook[-1]['min']
with codecs.open("results_ga.txt", "w", "utf-8-sig") as sys.stdout:
    print_results(elitist, costs_ga, PRIOS, workshop_names, participant_names)
sys.stdout = stdout

print(f"Costs GA: {logbook[-1]['min']} vs BruteForce: {costs_bf}")
