from genetic_alg import GeneticAlgorithm

ga = GeneticAlgorithm(population_size=6, generations=2, shape_file=None)
while ga.generation_count <= ga.generations:
    print(f"Generation {ga.generation_count}")
    ga.train_population(training_steps=1)
    ga.evaluate_population(episodes=1, steps=100, verbose=1)
    ga.evolve()
