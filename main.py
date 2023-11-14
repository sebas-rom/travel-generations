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
MAX_GENERATIONS = 10
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

def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             hof=None, verbose=__debug__):
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    ... (rest of the function remains unchanged)

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    best_individuals = []  # List to store the best individual of each generation
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(population)
        hof_size = len(hof.items)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        #print(logbook.stream)
        print("verbose")

    # Begin the generational process
    for gen in range(0, ngen):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population)-hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hof is not None:
            offspring.extend(hof.items)
            hof.update(offspring)
    
        # Replace the current population by the offspring
        population[:] = offspring

        # Append the best individual of the current generation to the list
        #best = hof.items[0]
        best = tools.selBest(population, 1)[0].copy()
        best_individuals.append(best)
       
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            route = " ==> ".join([cities[i][0] for i in best])# Collect minimum and average fitness values
            total_distance = calculate_total_distance(best)[0]
            print(f"Generation {gen} Route: {route}\nFitness value: {total_distance}\n")

    return population, logbook, best_individuals


def create_toolbox():
    if 'FitnessMin' in creator.__dict__:
        del creator.FitnessMin

    if 'Individual' in creator.__dict__:
        del creator.Individual
        
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
    hof = tools.HallOfFame(hall_of_fame_size)
    
    toolbox = create_toolbox()
    
    
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    
    stats.register("min", np.min)
    
    # Determinate if the algorithm will use elitism or not based on the hall of fame size
    elitism = "with" if hall_of_fame_size == 30 else "without"
    print(f"Running genetic algorithm {elitism} elitism...\n ")
    pop = toolbox.population(n=POPULATION_SIZE)
    pop, log,best_individuals = eaSimple(pop, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                       stats=stats, hof=hof, verbose=True)
        
    best_individual_original = best_individuals[0]
    best_individual_of_all = best_individuals[MAX_GENERATIONS-1]
      
    min_fitness_values = log.select("min")
    mean_fitness_values = log.select("avg")
        

    print("Best individual in the original population:")
    total_distance_o = calculate_total_distance(best_individual_original)[0]
    print("best individual original: ", best_individual_original)
    route_o = " ==> ".join([cities[i][0] for i in best_individual_original])
    print(f"Route: {route_o}\nDistance: {total_distance_o}")

    print("Best individual after all generations:")

    total_distance = calculate_total_distance(best_individual_of_all)[0]
    print("best individual: ", best_individual_of_all)
    route = " ==> ".join([cities[i][0] for i in best_individual_of_all])
    print(f"Route: {route}\nDistance: {total_distance}")
    

    
    
    print("\nPath obtained using the best individual from the original population")
    plot_route(cities=cities,best=best_individual_original, filename=generate_file_name(plot="route_original",elitism=elitism, random_seed=random_seed))
    print("=====================================================================================================")
    print("\nPath obtained using the best individual from all generations including the original population")
    plot_route(cities=cities,best=best_individual_of_all, filename=generate_file_name(plot="route",elitism=elitism, random_seed=random_seed))
    print("=====================================================================================================")
    print("\nPlot of the min and average fitness over generations")
    plot_Min_Avg_fitness(min_fitness_values=min_fitness_values, mean_fitness_values=mean_fitness_values,filename=generate_file_name(plot="statistics",elitism=elitism, random_seed=random_seed))
    print("=====================================================================================================")
    
def generate_file_name(plot,elitism, random_seed):
    filename=""
    if plot=="route":
        filename="Best route"
    elif plot=="route_original":
        filename="Best original route"
    elif plot=="statistics":
        filename="Min and Average Fitness"
    if elitism:
        filename += " with elitism"
    else:
        filename += " without elitism"
    if random_seed:
        filename += " with random seed"
    else:
        filename += " with seed 42"
    filename += ".png"
    return filename

def plot_Min_Avg_fitness(min_fitness_values, mean_fitness_values,filename):
    
    plt.clf()
    plt.plot(min_fitness_values, color='red', label='Min Fitness')
    plt.plot(mean_fitness_values, color='green', label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title(filename)
    plt.savefig(filename)
    plt.legend()
    plt.show()  
    
def plot_route(cities,best, filename):
    
    plt.clf()
    plt.scatter(*zip(*[city[1] for city in cities]), marker='.', color='red')
    locs = [cities[i][1] for i in best]
    locs.append(locs[0])
    plt.plot(*zip(*locs), linestyle='-', color='blue')
    # Annotate city numbers
    numero_ciudades = len(cities)
    for i in range(numero_ciudades):
        plt.annotate(cities[i][0], (cities[i][1][0], cities[i][1][1]), ha='left', va='bottom')  
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(filename)
    plt.savefig(filename)
    plt.show()
    plt.clf()
        
def main():
    # Run the genetic algorithm withour elitism and without random seed
    TSP(elitism=False, random_seed=False)
    
    # Run the genetic algorithm with elitism and without random seed
    
    TSP(elitism=True, random_seed=False)
    
    
    #TSP(elitism=False, random_seed=True)
    #TSP(elitism=True, random_seed=True)


if __name__ == '__main__':
    main()
