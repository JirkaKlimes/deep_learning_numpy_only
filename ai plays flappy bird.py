from games.flappy_bird import Game
from deep_learning import Agent, Layer, Population
from keyboard import is_pressed
import numpy as np



size = (800, 400)
population_size = 100

rate = 1
scale = 0.3
pool_size = 4
method = Population.RULETTE_WHEEL
include_parents = True
mutated_layers = Agent.RANDOM

l1 = Layer(3, 4, activation=Layer.RELU)
l2 = Layer(4, 2, activation=Layer.SOFTMAX)

agent = Agent([l1, l2])

population = Population(agent, population_size)

population = Population(file_name='birds.npy')
population.evolution_settings(include_parents, rate, scale, pool_size, method, population_size, mutated_layers=mutated_layers)

game = Game(size)
best_fitness = 0
gen = 0
while True:
    gen += 1
    best_gen_fitness = 0
    playing = True
    game.restart(population_size)
    while playing:
        playing = False
        game.update()
        for bird, agent in zip(game.birds, population.agents):
            if not bird.is_alive: continue
            playing = True
            by = bird.y/size[1]
            py = game.closset_pipe[1]/size[1]
            dist_to_pipe = (game.closset_pipe[0] - game.birds_x)/size[0]
            agent.push_forward([by, py, dist_to_pipe])
            jump = np.argmax(agent.outputs)
            bird.update(jump)
            agent.fitness = bird.increments
            if agent.fitness > best_fitness:
                best_fitness = agent.fitness
            if agent.fitness > best_gen_fitness:
                best_gen_fitness = agent.fitness
            if is_pressed('q'): quit()
    print(f'Gen: {gen} | Best fitness: {best_fitness} | Best gen fitness: {best_gen_fitness}')
    population.save(f'birds.npy', rewrite=True)
    population.evolve()
