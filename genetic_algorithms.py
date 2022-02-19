from re import L
from deep_learning import NeuralNetwork, Layer
import os
import math
import random
import numpy as np
from pathlib import Path


class Agent(NeuralNetwork):

    def __init__(self, layers, idn=None):
        super().__init__(layers)

        self.fitness = 0
        if idn is None:
            idn = id(self)
        self.idn = idn
        self.age = 1
        self.n_mutations = 0
        self.survival_prob = 0

    def copy(self):
        new_layers = []
        for layer in self.layers:
            new_layer = layer.copy()
            new_layers.append(new_layer)
        agent_copy = Agent(new_layers)
        return agent_copy


class Population():

    RULETTE_WHEEL = 'roulette wheel'
    TOP_X = 'top x of poulation'

    SQUARED = 'squared'


    def __init__(self, agent=None, size=None, unique=True, file_name=None):

        if file_name is not None:
            path = Path(f'{Path.cwd()}\populations\{file_name}')
            if not path.exists():
                print('ERROR WHILE LOADING POPULATION\n')
                print(f'Path: "{path}" doesn\'t exists!\n')
                return
            self.population = list(np.load(path, allow_pickle=True))
            self.size = len(self.population)
            return

        self.size = size
        self.population = []
        for i in range(size):
            new_agent = agent.copy()
            if unique:
                new_agent.mutate()
            new_agent.idn = i
            self.population.append(new_agent)
                
        self.selection_size = max(5, int(len(self.population)*0.1))
    

    def fitness_func(self, fitness, func):
        if func == self.SQUARED:
            return math.pow(fitness, 2)


    def calc_survival_prob(self, func=SQUARED):
        fitness_sum = 0
        for agent in self.population:
            fitness_sum += self.fitness_func(agent.fitness, func=func)
        if fitness_sum == 0:
            for agent in self.population:
                agent.survival_prob = 1/self.size
            return
        if fitness_sum == 0:
            fitness_sum = 1
        for agent in self.population:
            agent.survival_prob = self.fitness_func(agent.fitness, func=func) / fitness_sum


    def create_selection_pool(self, size=None, method=RULETTE_WHEEL):
        if size is None:
            size = self.selection_quantity

        self.calc_survival_prob()

        if method == self.RULETTE_WHEEL:
            mating_pool = []
            for idx in range(size):
                picked = self.pick_agent()
                mating_pool.append(picked)
            return mating_pool
        
        if method == self.TOP_X:
            mating_pool = sorted(self.population, key=lambda a: a.survival_prob, reverse=True)[:size]
            return mating_pool


    def pick_agent(self):
        idx = 0
        rand = random.random()
        while rand > 0:
            rand -= self.population[idx].survival_prob
            idx += 1
        idx -= 1
        return self.population[idx]

    def evolve(self, mutation_rate=0.7, mutation_scale=0.1, selection_size=None, selection_method=RULETTE_WHEEL, poulation_size=None, include_parents=True):
        if poulation_size is None:
            poulation_size = self.size
        
        if selection_size is None:
            selection_size = self.selection_size
        
        self.calc_survival_prob()

        new_population = []

        survials = self.create_selection_pool(size=selection_size, method=selection_method)
        if include_parents:
            for agent in survials:
                new_agent = agent.copy()
                new_agent.age = agent.age
                new_agent.n_mutations = agent.n_mutations
                new_population.append(new_agent)

        for agent in new_population:
            agent.mutate(mutation_rate, scale=mutation_scale, repeat=True)

        while len(new_population) < poulation_size:
            mutant = random.choice(survials).copy()
            mutant.mutate(mutation_rate, scale=mutation_scale)
            mutant.n_mutations += 1
            new_population.append(mutant)
        
        for agent in new_population:
            agent.age += 1
        self.population = new_population
    
    def save(self, file_name='population.npy', rewrite=False):
        path = Path(f'{Path.cwd()}\populations')
        if Path(f'{path}\{file_name}').exists():
            if not rewrite:
                print('ERROR WHILE SAVING POPULATION\n')
                print(f'File: "{path}\{file_name}" already exists!\n')
                return
            if rewrite: Path(f'{path}\{file_name}').unlink()
        path.mkdir(exist_ok=True)
        path = Path(f'{path}\{file_name}')
        np.save(path, np.array(self.population), allow_pickle=True)


from snake import Snake, Vis

size = 10
n_games = 3
vis_enabled = True
population_size = 300



max_steps = 200
rate = 0.05
scale = 1
selection_size = 5
method = Population.RULETTE_WHEEL


# l1 = Layer(8, 8, activation=Layer.RELU)
# l2 = Layer(8, 8, activation=Layer.RELU)
# l3 = Layer(8, 4, activation=Layer.SOFTMAX)

# agent = Agent([l1, l2, l3])

# poulation = Population(agent, population_size)

poulation = Population(file_name='snakes.npy')

snakes = [Snake(size) for _ in range(population_size)]
if vis_enabled: vis = Vis(snakes[:100])


best_fitness = -1000
best_score = 0

gen = 0
while True:
    gen += 1
    
    for game_idx in range(n_games):
        i = max_steps
        while i > 0:
            snake_playing = False
            for idx, snake, agent in zip(range(population_size), snakes, poulation.population):
                if snake.gameover: continue
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
                agent.fitness = -snake.steps_taken + snake.score*100
                if agent.fitness > best_fitness:
                    best_fitness = agent.fitness
            if vis_enabled: vis.update()
            if not snake_playing:
                break
            i -= 1
        snakes = [Snake(size) for _ in range(population_size)]
        if vis_enabled: vis.snakes = snakes[:100]
        os.system('cls')
        print(f'Gen: {gen} | Best fitness: {best_fitness} | Game: {game_idx+1}/{n_games} | Best score: {best_score}')
        poulation.save(f'snakes.npy', rewrite=True)
    poulation.evolve(rate, scale, selection_size, method, population_size)