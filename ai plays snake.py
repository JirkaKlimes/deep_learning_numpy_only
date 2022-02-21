from games.snake import Snake, Vis
from keyboard import is_pressed
import os
import numpy as np

from deep_learning import Agent, Layer, Population


size = 10
n_games = 1
vis_enabled = False
population_size = 100


max_steps = 30
rate = 1
scale = 0.1
pool_size = 10
method = Population.TOP_X
include_parents = False

l1 = Layer(8, 10, activation=Layer.RELU)
l2 = Layer(10, 10, activation=Layer.RELU)
l3 = Layer(10, 4, activation=Layer.SOFTMAX)

agent = Agent([l1, l2, l3])
poulation = Population(agent, population_size)


# poulation = Population(file_name='snakes.npy')
poulation.evolution_settings(include_parents=include_parents, mutation_rate=rate, mutation_scale=scale,
                             pool_size=pool_size, selection_method=method, population_size=population_size)


snakes = [Snake(size) for _ in range(population_size)]
if vis_enabled:
    vis = Vis(snakes[:100])


best_fitness = -1000
best_score = 0

gen = 0
while True:
    gen += 1
    best_gen_score = -10000
    for game_idx in range(n_games):
        i = max_steps
        while i > 0:
            snake_playing = False
            for idx, snake, agent in zip(range(population_size), snakes, poulation.agents):
                if snake.gameover:
                    continue
                agent.push_forward(snake.get_steps())
                direction = np.argmax(agent.outputs)
                match direction:
                    case 0:
                        snake.set_dir(snake.UP)
                    case 1:
                        snake.set_dir(snake.LEFT)
                    case 2:
                        snake.set_dir(snake.DOWN)
                    case 3:
                        snake.set_dir(snake.RIGHT)
                if not snake.gameover:
                    score = snake.update()
                if score is False or True:
                    snake_playing = True
                if score is True:
                    i = max_steps
                if snake.score > best_score:
                    best_score = snake.score
                if snake.score > best_gen_score:
                    best_gen_score = snake.score
                agent.fitness = snake.steps_taken + snake.score*100
                if agent.fitness > best_fitness:
                    best_fitness = agent.fitness
            if vis_enabled:
                vis.update()
            if not snake_playing:
                break
            i -= 1
            if is_pressed('q'):
                quit()
        snakes = [Snake(size) for _ in range(population_size)]
        if vis_enabled:
            vis.snakes = snakes[:100]
        os.system('cls')
        print(f'Gen: {gen} | Best fitness: {best_fitness} | Game: {game_idx+1}/{n_games} | Best score: {best_score}/{best_gen_score}')

    poulation.save(f'snakes.npy', rewrite=True)
    poulation.evolve()


# import cProfile

# with cProfile.Profile() as pr:
#     main()

# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)

# stats.dump_stats('stats.prof')
