import random
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
from geopy.distance import geodesic
import time
# Constants
POPULATION_SIZE = 300
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 200
HALL_OF_FAME_SIZE = 5
RANDOM_SEED = 42

# Define cities and coordinates
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

def create_toolbox():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(cities)), len(cities))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(cities))
    toolbox.register("select", tools.selTournament, tournsize=2)

    # Register the evaluation function
    toolbox.register("evaluate", calculate_total_distance)

    return toolbox


def calculate_total_distance(individual):
    distance = 0
    for i in range(len(individual)):
        from_city = cities[individual[i-1]][1]
        to_city = cities[individual[i]][1]
        distance += geodesic(from_city, to_city).km
    return distance,
def TSP(elitism=False, random_seed=False):
    
    if random_seed==True:
        random.seed(time.time())
        print("Running genetic algorithm with random seed...\n")
    else:
        random.seed(RANDOM_SEED)
        print("Running genetic algorithm with random seed= 42\n")
    
    hall_of_fame_size = 30 if elitism else 5
            
    #toolbox = create_toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(cities)), len(cities))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(cities))
    toolbox.register("select", tools.selTournament, tournsize=2)

    # Register the evaluation function
    toolbox.register("evaluate", calculate_total_distance)
    
    hof = tools.HallOfFame(hall_of_fame_size)

    pop = toolbox.population(n=POPULATION_SIZE)
    min_fitness_values = []
    mean_fitness_values = []
    
    # Determinate if the algorithm will use elitism or not based on the hall of fame size
    elitism = "with" if hall_of_fame_size == 30 else "without"
    print(f"Running genetic algorithm {elitism} elitism...\n ")

    for generation in range(MAX_GENERATIONS):
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=1,
                                       stats=None, halloffame=hof, verbose=False)
        
        best_individual = tools.selBest(pop, 1)[0]
        total_distance = calculate_total_distance(best_individual)[0]
        route = " ==> ".join([cities[i][0] for i in best_individual])
        
        print(f"Generation {generation + 1} Route: {route}\nFitness value: {total_distance}\n")
        
        # Collect minimum and average fitness values
        min_fitness = min(ind.fitness.values[0] for ind in pop)
        mean_fitness = sum(ind.fitness.values[0] for ind in pop) / len(pop)
        min_fitness_values.append(min_fitness)
        mean_fitness_values.append(mean_fitness)
        if generation == 0:
            best_individual_original = hof[0]

    print("Best individual after all generations:")
    best_individual = hof[0]
    total_distance = calculate_total_distance(best_individual)[0]
    print("best individual: ", best_individual)
    route = " ==> ".join([cities[i][0] for i in best_individual])
    print(f"Route: {route}\nDistance: {total_distance}")
    
    filename_for_best_route_original_population = f"best_original_route_{elitism}_elitism"
    if random_seed:
        filename_for_best_route_original_population += "_with_radom_seed"
    else:
        filename_for_best_route_original_population += "_with_seed_42"
    filename_for_best_route_original_population += ".png"
    
    
    filename_for_best_route = f"best_route_{elitism}_elitism"
    if random_seed:
        filename_for_best_route += "_with_random_seed"
    else:
        filename_for_best_route += "_with_seed_42"
    filename_for_best_route += ".png"
    
    
    filename_for_statistics = f"statistics_{elitism}_elitism"
    if random_seed:
        filename_for_statistics += "_with_radom_seed"
    else:
        filename_for_statistics += "_with_seed_42"
    filename_for_statistics += ".png"
    
    
    print("\nPath obtained using the best individual from the original population")
    plot_route(cities=cities,best_individual=best_individual_original, filename=filename_for_best_route_original_population)
    print("=====================================================================================================")
    print("\n Path obtained using the best individual from all generations including the original population")
    plot_route(cities=cities,best_individual=best_individual, filename=filename_for_best_route)
    print("=====================================================================================================")
    print("Plot of the min and average fitness over generations")
    plot_Min_Avg_fitness(min_fitness_values=min_fitness_values, mean_fitness_values=mean_fitness_values,filename=
                         filename_for_statistics)
    print("=====================================================================================================")
    
def main():
    # Run the genetic algorithm withour elitism and without random seed
    TSP(elitism=False, random_seed=False)
    
    # Run the genetic algorithm with elitism and without random seed
    
    TSP(elitism=True, random_seed=False)
    #TSP(elitism=False, random_seed=True)
    #TSP(elitism=True, random_seed=True)

def plot_Min_Avg_fitness(min_fitness_values, mean_fitness_values,filename):
    plt.clf()
    plt.plot(min_fitness_values, color='red', label='Min Fitness')
    plt.plot(mean_fitness_values, color='green', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average Fitness over Generations')
    plt.savefig(filename)
    plt.legend()
    plt.show()  
    
def plot_route(cities,best_individual, filename):
    plt.clf()
    plt.scatter(*zip(*[city[1] for city in cities]), marker='.', color='red')
    locs = [cities[i][1] for i in best_individual]
    locs.append(locs[0])
    plt.plot(*zip(*locs), linestyle='-', color='blue')
    # Annotate city numbers
    numero_ciudades = len(cities)
    for i in range(numero_ciudades):
        plt.annotate(cities[i][0], (cities[i][1][0], cities[i][1][1]), ha='left', va='bottom')  
    
    plt.title("Route")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(filename)
    plt.savefig(filename)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    main()
