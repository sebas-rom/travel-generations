import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from deap import algorithms, base, creator, tools
from geopy.distance import geodesic

# Ciudades y coordenadas
cities = [
    ("Seattle", (47.608013, -122.335167)),
    ("Boise", (43.616616, -116.200886)),
    ("Everett", (47.967306, -122.201399)), 
    ("Pendleton", (45.672075, -118.788597)),
    ("Biggs", (45.669846, -120.832841)),
    ("Portland", (45.520247, -122.674194)),
    ("Twin Falls", (42.570446, -114.460255)),
    ("Bend", (44.058173, -121.315310)),
    ("Spokane", (47.657193, -117.423510)),
    ("Grant Pass", (42.441561, -123.339336)),
    ("Burns", (43.586126, -119.054413)),
    ("Eugene", (44.050505, -123.095051)),
    ("Lakeview", (42.188772, -120.345792)),
    ("Missoula", (46.870105, -113.995267))
]

# Parámetros del algoritmo genético
POPULATION_SIZE = 300
P_CROSSOVER = 0.9 
P_MUTATION = 0.1
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 5
RANDOM_SEED = 42

# Crea individuos y población
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(cities)), len(cities))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(cities))
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", lambda x: calc_total_distance(x))

# Hall of Fame para guardar los mejores individuos
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Función para calcular la distancia total de una ruta
def calc_total_distance(individual):
    distance = 0
    for i in range(len(individual)):
        from_city = cities[individual[i-1]][1]
        to_city = cities[individual[i]][1]
        distance += geodesic(from_city, to_city).km
    return distance,

def main():
    
    random.seed(RANDOM_SEED)

    pop = toolbox.population(n=POPULATION_SIZE)

    # Algoritmo sin elitismo
    print("Ejecutando algoritmo genético sin elitismo...")

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                   ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    best = hof.items[0]
    print("Mejor individuo = ", best)
    print("Distancia = ", calc_total_distance(best)[0])

    # Graficar ruta
    x, y = zip(*[cities[i][1] for i in best])
    plt.plot(x, y, 'bo-')
    plt.plot(x[0], y[0], 'r*')
    plt.title("Ruta inicial")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.savefig("initial_route.png")
    plt.clf()

    # Algoritmo con elitismo
    print("Ejecutando algoritmo genético con elitismo...")
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, 
                                   ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    best = hof.items[0]
    print("Mejor individuo = ", best)
    print("Distancia = ", calc_total_distance(best)[0])

    # Graficar mejor ruta 
    x, y = zip(*[cities[i][1] for i in best])
    plt.plot(x, y, 'bo-')
    plt.plot(x[0], y[0], 'r*') 
    plt.title("Mejor ruta")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.savefig("best_route.png")

if __name__ == "__main__":
    main()