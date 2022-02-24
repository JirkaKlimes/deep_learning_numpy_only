from games.snake import Snake, Vis
from keyboard import is_pressed
import os
import numpy as np

from deep_learning import Agent, Layer, NeuralNetwork, Population


size = 8
n_games = 1
vis_enabled = False
population_size = 100

max_steps = 30
rate = 0.5
scale = 0.2
pool_size = 4
method = Population.TOP_X
include_parents = True
mutated_layers = NeuralNetwork.ALL

# l1 = Layer(8, 4, activation=Layer.RELU)
# l2 = Layer(4, 4, activation=Layer.RELU)


# agent = Agent([l1, l2])
# population = Population(agent, population_size)

population = Population(file_name='snakes.npy')

population.evolution_settings(include_parents=include_parents, 
                              pool_size=pool_size, selection_method=method, 
                              population_size=population_size)

population.mutation_settings(mutation_rate=rate, mutation_scale=scale, mutated_layers=mutated_layers)

population.expansion_settings(100, 0.1, 100, 0.5)

snakes = [Snake(size) for _ in range(population_size)]
if vis_enabled: vis = Vis(snakes[:vis_enabled])

while True:
    for game_idx in range(n_games):
        i = max_steps
        while i > 0:
            i -= 1
            snake_playing = False
            for idx, snake, agent in zip(range(population_size), snakes, population.agents):
                if snake.gameover:
                    continue
                agent.push_forward(snake.get_steps())
                direction = np.argmax(agent.outputs)
                match direction:
                    case 0: snake.set_dir(snake.UP)
                    case 1: snake.set_dir(snake.LEFT)
                    case 2: snake.set_dir(snake.DOWN)
                    case 3: snake.set_dir(snake.RIGHT)
                if not snake.gameover:
                    score = snake.update()
                if score is False or True:
                    snake_playing = True
                if score is True: i = max_steps
                agent.fitness = -snake.steps_taken + snake.score*100
            if vis_enabled:
                vis.update()
            if not snake_playing:
                break
            if is_pressed('q'): quit()
        snakes = [Snake(size) for _ in range(population_size)]
        if vis_enabled: vis.snakes = snakes[:vis_enabled]

    population.save(f'snakes.npy', rewrite=True)
    population.evolve()
