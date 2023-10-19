import array
import random
import utm
import numpy as np
from geopy import distance
from pyproj import Proj
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

random.seed(42)

# (lat, lon)
seattle = (47.608013, -122.335167)
boise = (43.616616, -116.200886)
everett = (47.967306, -122.201399)
pendleton = (45.672075, -118.788597)
biggs = (45.669846, -120.832841)
portland = (45.520247, -122.674194)
twin_falls = (42.570446, -114.460255)
bend = (44.058173, -121.315310)
spokane = (47.657193, -117.423510)
grant_pass = (42.441561, -123.339336)
burns = (43.586126, -119.054413)
eugene = (44.050505, -123.095051)
lakeview = (42.188772, -120.345792)
missoula = (46.870105, -113.995267)


p = Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)

x1, y1 = p(-114.460255, 42.570446)  # twin_falls
x2, y2 = p(-122.335167, 47.608013)  # seattle

xCoords = [x1, x2]
yCoords = [y1, y2]

# print(np.sqrt(np.diff(xCoords) * 2 + np.diff(yCoords) * 2) / 1000)

city_mapping = {

    0: ["seattle", (47.608013, -122.335167)],
    1: ["boise", (43.616616, -116.200886)],
    2: ["everett", (47.967306, -122.201399)],
    3: ["pendleton", (45.672075, -118.788597)],
    4: ["biggs", (45.669846, -120.832841)],
    5: ["portland", (45.520247, -122.674194)],
    6: ["twin_falls", (42.570446, -114.460255)],
    7: ["bend", (44.058173, -121.315310)],
    8: ["spokane", (47.657193, -117.423510)],
    9: ["grant_pass", (42.441561, -123.339336)],
    10: ["burns", (43.586126, -119.054413)],
    11: ["eugene", (44.050505, -123.095051)],
    12: ["lakeview", (42.188772, -120.345792)],
    13: ["missoula", (46.870105, -113.995267)]

}

# print(city_mapping[0])
# print(city_mapping[0][1])
# print(city_mapping[0][1][0])


def calc_distancias(ciudades):
    number_cities = len(ciudades)
    distances = [[0] * number_cities for _ in ciudades]
    for i in range(number_cities):
        for j in range(i + 1, number_cities):
            distancia = distance.distance(ciudades[j][1], ciudades[i][1]).km
            distances[i][j] = distancia
            distances[j][i] = distancia
    return distances


def transformar_coordenadas(ciudades):
    coordenadas = []
    number_cities = len(ciudades)
    for i in range(number_cities):
        lat = ciudades[i][1][0]
        lon = ciudades[i][1][1]
        x = utm.from_latlon(lat, lon)
        coordenadas.append([x[0], x[1]])
    return coordenadas


COORDENADAS = transformar_coordenadas(city_mapping)
#print(COORDENADAS)
# print("\n\n")
# print(COORDENADAS[0][0])
DISTANCIAS = calc_distancias(city_mapping)
NUMERO = len(city_mapping)
# print(city_mapping[0])

NUM_GENERATIONS = 200
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.1

individual = list(range(NUMERO))
individual = random.sample(individual, len(individual))


# print(individual)


def distancia_particular(individual: list) -> float:
    # get distance between first and last city
    distancia = DISTANCIAS[individual[0]][individual[0]]
    # add all other distances
    for i in range(NUMERO - 1):
        distancia += DISTANCIAS[individual[i]][individual[i + 1]]
    return distancia


creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Create operator to shuffle the cities
toolbox.register('randomOrder', random.sample, range(NUMERO), NUMERO)
# Create initial random individual operator
toolbox.register('individualCreator', tools.initIterate, creator.Individual, toolbox.randomOrder)
# Create random population operator
toolbox.register('populationCreator', tools.initRepeat, list, toolbox.individualCreator)


def distancia_particular_tuple(individual) -> tuple:
    return distancia_particular(individual),


toolbox.register('evaluate', distancia_particular_tuple)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxOrdered)
toolbox.register('mutate', tools.mutShuffleIndexes, indpb=1.0 / len(individual))

population = toolbox.populationCreator(n=POPULATION_SIZE)
#########################################################################
HALL_OF_FAME_SIZE = 30
HALL_OF_FAME_SIZE1 = 5
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('avg', np.mean)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields

invalid_individuals = [ind for ind in population if not ind.fitness.valid]
fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
for ind, fit in zip(invalid_individuals, fitnesses):
    ind.fitness.values = fit

hof.update(population)
hof_size = len(hof.items)

record = stats.compile(population)
logbook.record(gen=0, nevals=len(invalid_individuals), **record)
print(logbook.stream)

for gen in range(1, NUM_GENERATIONS + 1):
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population) - hof_size)

    # Vary the pool of individuals
    offspring = algorithms.varAnd(offspring, toolbox, P_CROSSOVER, P_MUTATION)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # add the best back to population:
    offspring.extend(hof.items)

    # Update the hall of fame with the generated individuals
    hof.update(offspring)

    # Replace the current population by the offspring
    population[:] = offspring

    # Append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    if gen == NUM_GENERATIONS:
        best = hof.items[0]
        print('Best Individual = ', best)
        print('Best Fitness = ', best.fitness.values[0])
        plt.scatter(*zip(*COORDENADAS), marker='.', color='red')
        locs = [COORDENADAS[i] for i in best]
        locs.append(locs[0])
        plt.plot(*zip(*locs), linestyle='-', color='blue')
        numero_ciudades = len(COORDENADAS)
        for i in range(numero_ciudades):
            plt.annotate(i, (COORDENADAS[i][0], COORDENADAS[i][1]))
        plt.figure(1)


        # plot genetic flow statistics:
        minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
        plt.figure(2)
        plt.plot(minFitnessValues, color='red')
        plt.plot(meanFitnessValues, color='green')
        plt.xlabel('Generation')
        plt.ylabel('Min / Average Fitness')
        plt.title('Min and Average fitness over Generations')
        # show both plots:
        plt.show()