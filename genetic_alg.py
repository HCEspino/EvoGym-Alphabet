from robots import Organism, random_from_shape
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class GeneticAlgorithm():
    def __init__(self, population_size=16, generations=100, shape_file=None):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_organism = None
        self.best_fitness = 0

        self.generation_count = 1

        self.init_population(shape_file)

    def init_population(self, shape_file):
        '''
        Initializes the population with random organisms
        '''
        for _ in range(self.population_size):
            body, connections = random_from_shape(shape_file)
            self.population.append(Organism(body, connections))

    def train_population(self, training_steps=50000):
        '''
        Trains the entire population
        '''
        for organism in self.population:
            organism.train(training_steps)

    def evaluate_population(self, episodes=10, steps=500, verbose=True):
        '''
        Evaluates the entire population
        '''
        for organism in self.population:
            organism.evaluate(episodes, steps)

        if verbose:
            print(f"Generation {self.generation_count}")
            print(f"Best fitness: {self.best_fitness}, Average fitness: {np.mean([x.fitness for x in self.population])}")
            print(f"Population size: {len(self.population)}")

    def evolve(self):
        '''
        Evolves the population, selecting the best organisms and breeding them, then mutating the offspring
        '''
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_organism = self.population[0]
        self.best_fitness = self.best_organism.fitness

        # Videos of the top 3 organisms
        self.population[0].save_to_video(video_name=f"gen_{self.generation_count}_rank_1")
        self.population[1].save_to_video(video_name=f"gen_{self.generation_count}_rank_2")
        self.population[2].save_to_video(video_name=f"gen_{self.generation_count}_rank_3")

        # New population is top 50% of the current population
        new_population = self.population[:self.population_size // 2]

        # Breed the top 50% of the population
        pairs = []
        while len(new_population) + len(pairs) < self.population_size:

            # Select parents based on fitness
            parent1 = np.random.choice(new_population, p=softmax([x.fitness for x in new_population]))
            # Ensure parent2 is not the same as parent1
            other = [x for x in new_population if x != parent1]
            parent2 = np.random.choice(other, p=softmax([x.fitness for x in other]))

            pairs.append((parent1, parent2))

        # Create offspring from pairs
        for pair in pairs:
            new_population.append(pair[0].create_offspring(pair[1]))

        self.generation_count += 1



