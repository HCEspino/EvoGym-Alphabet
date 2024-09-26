from genetic_alg import GeneticAlgorithm

ga = GeneticAlgorithm(population_size=16, generations=30, shape_file="D.npy")
while ga.generation_count <= ga.generations:
    print(f"Generation {ga.generation_count}")
    ga.train_population(training_steps=50000)
    ga.evaluate_population(episodes=1, steps=500, verbose=1)
    ga.evolve()
