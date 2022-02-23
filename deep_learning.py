import enum
import numpy as np
import math
import pygame
import time
from keyboard import is_pressed
from pathlib import Path
import cv2 as cv
import random

class Layer:
    
    SIGMOID = 'sigmoid'
    RELU = 'relu'
    SOFTMAX = 'softmax'
    SHARP_SIGMOID = 'sharpsigmoid'


    NONE = 'none'
    MIN_MAX = 'minmax'
    DIV_BY_MAX = 'divbymax'

    def __init__(self, inputs, neurons, activation=SIGMOID, normalization=NONE, rand_weights=True):
        if rand_weights: self.weights = np.random.standard_normal((inputs, neurons))
        else: self.weights = np.zeros((inputs, neurons))

        self.neurons = neurons
        self.inputs = inputs

        self.biases = np.zeros((1, neurons))

        self.activation = activation
        self.normalization = normalization

    def push_forward(self, inputs):
        inputs = np.array(inputs)
        if 'array' not in str(type(inputs[0])):
            inputs = inputs.reshape((1, -1))
        self.input_values = inputs

        match self.normalization:
            case self.MIN_MAX:
                pass
            case self.DIV_BY_MAX:
                absolute = np.abs(inputs)
                maximum = np.max(absolute, axis=1, keepdims=True)
                inputs = np.divide(inputs, maximum)
            case self.NONE:
                pass
                
        outputs = np.dot(inputs, self.weights) + self.biases

        match self.activation:
            case self.SIGMOID:
                self.outputs = np.divide(1, 1+np.exp(-outputs))
            case self.RELU:
                self.outputs = np.maximum(0, outputs)
            case self.SOFTMAX:
                e = np.exp(outputs)
                e = np.divide(e, np.max(e, axis=1, keepdims=True))
                self.outputs = np.divide(e, np.sum(e, axis=1, keepdims=True))
            case self.SHARP_SIGMOID:
                self.outputs = np.minimum(1, np.maximum(0, outputs))

    
    def mutate(self, rate, n_neurons=0):
        neurons = range(0, len(self.biases[0]))
        n_neurons = n_neurons if n_neurons > 0 else random.randint(0, len(self.biases[0]-1))
        neurons_to_mutate = random.sample(neurons, k=min(len(self.biases[0]), n_neurons))
        for n in neurons_to_mutate:
            self.weights.T[n] += np.random.uniform(-rate, rate, self.weights.T[n].shape[0])
            self.biases[0][n] += random.uniform(-rate, rate)

    def copy(self):
        copied = Layer(self.inputs, self.neurons, activation=self.activation, normalization=self.normalization)
        copied.weights = np.array(self.weights, copy=True, order='K')
        copied.biases = np.array(self.biases, copy=True, order='K')
        return copied

class NeuralNetwork:

    RANDOM = 'random'
    ALL = 'all'

    class Visualization:

        color_background = (10, 10, 10)

        color_neuron_positive = (237, 144, 31)
        color_neuron_negative = (0, 128, 255)
        color_neuron_inner = (255, 255, 255)
        color_inputs_outer = (255, 255, 255)

        color_connection_negative = (237, 144, 31)
        color_connection_positive = (0, 128, 255)

        radius_neuron_ratio = 80
        thickness_neuron = 2
        thickness_connection = 1

        black_color_threshold = 30

        def __init__(self, layers, screen_size=(1600, 800), left_layer_ofset=0.1, window_name='Neural Network'):
            self.window_name = window_name
            self.layers = layers
            self.architecture = np.array([[l.inputs, l.neurons] for l in layers], dtype=object)
            self.width = screen_size[0]
            self.height = screen_size[1]
            self.screen_size = screen_size
            self.left_layer_ofset = int(left_layer_ofset*self.width)

            self.blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.blank = cv.rectangle(self.blank, (0, 0), (self.width, self.height), self.color_background, -1)
            self.layer_spacing = int((self.width - 2*self.left_layer_ofset) / (self.architecture.shape[0]))

            pygame.init()
            self.create_screen()

        def create_screen(self):
            self.screen = pygame.display.set_mode(self.screen_size)
            self.running = True
        
        def show_net(self, frame_time=0, update_background=True):
            if update_background: self.update_background()
            if not self.running:
                self.create_screen()
            bkg = pygame.surfarray.make_surface(np.rot90(self.background, k=3))
            self.screen.blit(bkg, (0,0))
            if frame_time < 0:
                pygame.display.flip()
                return
            elif frame_time == 0:
                while self.running:
                    pygame.display.flip()
                    self.handle_events()
                    if is_pressed('n'):
                        break
            else:
                t1 = time.time()
                while time.time() < t1 + frame_time:
                    pygame.display.flip()
                    self.handle_events()
                    if is_pressed('n'):
                        break
            
        def show_activations(self, frame_time, last_frame=False, hold=False, quit=False):
            if not self.running:
                self.create_screen()
            bkg = pygame.surfarray.make_surface(np.rot90(self.background, k=3))
            self.screen.blit(bkg, (0,0))

            if frame_time == 0 or hold:
                while self.running:
                    pygame.display.flip()
                    self.handle_events()
                    if is_pressed('n'):
                        break

            if frame_time > 0:
                start_time = time.time()
                while time.time() < start_time + frame_time and self.running:
                    pygame.display.flip()
                    self.handle_events()
            
            if last_frame and quit:
                self.running = False
                pygame.quit()
                
        def handle_events(self):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    quit()

        def update_background(self):
            self.architecture = np.array([[l.inputs, l.neurons] for l in self.layers], dtype=object)
            self.radius_neuron = min(int(self.height / (self.architecture.max()*2 + 2) / 2), int(self.width / (len(self.architecture)*2 + 2) / 2))
            self.radius_neuron = max(1, self.radius_neuron)

            self.layer_spacing = int((self.width - 2*self.left_layer_ofset) / (self.architecture.shape[0]))
            self.background = np.array(self.blank, copy=True, order='K')
            self.draw_weights()
            self.draw_neurons()
        
        def show_building_net(self, frame_time=-1):
            self.background = np.array(self.blank, copy=True, order='K')
            self.draw_weights(show=True, frame_time=frame_time)
            self.draw_neurons(show=True, frame_time=frame_time)


        def normalize(self, array):
            absolute_values = np.absolute(array)
            max_value = np.max(absolute_values)
            if max_value == 0:
                return array
            return np.divide(array, max_value)
            
        def draw_neurons(self, show=False, frame_time=-1):

            x = self.left_layer_ofset
            for layer_idx, layer in enumerate(self.layers):
                neuron_spacing = int(self.height / (layer.inputs+1))
                y = neuron_spacing
                for idx in range(layer.inputs):
                    if layer_idx == 0:
                        self.background = cv.circle(self.background, (x, y), self.radius_neuron, self.color_inputs_outer, self.thickness_neuron, lineType=cv.LINE_AA)
                    else:
                        color = self._get_bias_color(biases[idx])
                        if color:
                            self.background = cv.circle(self.background, (x, y), self.radius_neuron, color, self.thickness_neuron, lineType=cv.LINE_AA)
                            if show: self.show_net(frame_time=frame_time, update_background=False)
                        else:
                            self.background = cv.circle(self.background, (x, y), self.radius_neuron, self.color_inputs_outer, self.thickness_neuron, lineType=cv.LINE_AA)
                            if show: self.show_net(frame_time=frame_time, update_background=False)
                    self.background = cv.circle(self.background, (x, y), int(self.radius_neuron-self.thickness_neuron/2), self.color_background, -1, lineType=cv.LINE_AA)
                    if show: self.show_net(frame_time=frame_time, update_background=False)
                    y += neuron_spacing
                x += self.layer_spacing
                biases = layer.biases[0]
                if biases.max() != 0:
                    biases = self.normalize(biases)

            neuron_spacing = int(self.height / (layer.neurons+1))
            y = neuron_spacing
            for idx in range(layer.neurons):
                bias = biases[idx]
                color = self._get_bias_color(bias)
                if color:
                    self.background = cv.circle(self.background, (x, y), self.radius_neuron, color, self.thickness_neuron)
                    if show: self.show_net(frame_time=frame_time, update_background=False)
                else:
                    self.background = cv.circle(self.background, (x, y), self.radius_neuron, self.color_inputs_outer, self.thickness_neuron, lineType=cv.LINE_AA)
                    if show: self.show_net(frame_time=frame_time, update_background=False)
                self.background = cv.circle(self.background, (x, y), int(self.radius_neuron-self.thickness_neuron/2), self.color_background, -1, lineType=cv.LINE_AA)
                if show: self.show_net(frame_time=frame_time, update_background=False)            
                y += neuron_spacing

        def draw_weights(self, show=False, frame_time=-1):

            x = self.left_layer_ofset
            for layer in self.layers:
                weights = self.normalize(layer.weights)
                neuron_spacing = int(self.height / (layer.inputs+1))
                next_layer_spacing = int(self.height / (layer.neurons+1))
                y = neuron_spacing
                for input_idx in range(layer.inputs):
                    y2 = next_layer_spacing
                    x2 = x+self.layer_spacing
                    for neuron_idx in range(layer.neurons):
                        current_weight = weights[input_idx][neuron_idx]
                        weight_color = self._get_weight_color(current_weight)
                        if weight_color:
                            self.background = cv.line(self.background, (x, y), (x2, y2), weight_color, self.thickness_connection, cv.LINE_AA)
                            if show: self.show_net(frame_time=frame_time, update_background=False)
                        y2 += next_layer_spacing
                    y += neuron_spacing
                x += self.layer_spacing
        
        def _get_weight_color(self, weight):
            if weight < 0:
                weight = abs(weight)
                color = (min(255, int(self.color_connection_negative[0]*weight)), min(255, int(self.color_connection_negative[1]*weight)), min(255, int(self.color_connection_negative[2]*weight)))
            else:
                color = (min(255, int(self.color_connection_positive[0]*weight)), min(255, int(self.color_connection_positive[1]*weight)), min(255, int(self.color_connection_positive[2]*weight)))
            if max(color) < self.black_color_threshold: return False
            return color

        def _get_bias_color(self, bias):
            if bias < 0:
                bias = abs(bias)
                color = (min(255, int(self.color_neuron_negative[0]*bias)), min(255, int(self.color_neuron_negative[1]*bias)), min(255, int(self.color_neuron_negative[2]*bias)))
            else:
                color = (min(255, int(self.color_neuron_positive[0]*bias)), min(255, int(self.color_neuron_positive[1]*bias)), min(255, int(self.color_neuron_positive[2]*bias)))
            if max(color) < self.black_color_threshold: return False
            return color
        
        def fill_neurons(self, layer_idx, output):
            if not output:
                x = self.left_layer_ofset + self.layer_spacing * layer_idx
                layer = self.layers[layer_idx]
                neuron_spacing = int(self.height / (layer.inputs+1))
                y = neuron_spacing
                activations = layer.input_values
                if type(activations[0]) == type(np.empty(1)):
                    activations = activations[0]
                activations = self.normalize(activations)
                for neuron_idx in range(layer.inputs):
                    current_activation = activations[neuron_idx]
                    color = (min(255, int(self.color_neuron_inner[0]*current_activation)), min(255, int(self.color_neuron_inner[1]*current_activation)), min(255, int(self.color_neuron_inner[2]*current_activation)))
                    self.background = cv.circle(self.background, (x, y), int(self.radius_neuron-self.thickness_neuron/2), color, -1, lineType=cv.LINE_AA)
                    y += neuron_spacing

            else:
                x = self.left_layer_ofset + self.layer_spacing * (layer_idx+1)
                layer = self.layers[-1]
                neuron_spacing = int(self.height / (layer.neurons+1))
                y = neuron_spacing
                activations = layer.outputs
                if type(activations[0]) == type(np.empty(1)):
                    activations = activations[0]
                activations = self.normalize(activations)
                for neuron_idx in range(layer.neurons):
                    current_activation = activations[neuron_idx]
                    color = (min(255, int(self.color_neuron_inner[0]*current_activation)), min(255, int(self.color_neuron_inner[1]*current_activation)), min(255, int(self.color_neuron_inner[2]*current_activation)))
                    self.background = cv.circle(self.background, (x, y), int(self.radius_neuron-self.thickness_neuron/2), color, -1, lineType=cv.LINE_AA)
                    y += neuron_spacing

    def __init__(self, layers):
        self.layers = layers
        self.architecture = np.array([[l.inputs, l.neurons] for l in layers])
    
    visualization_enabled = False
    def enable_visualization(self, size=(1600, 800)):
        self.vis = self.Visualization(screen_size=size, layers=self.layers)
        self.visualization_enabled = True

    def push_forward(self, inputs, frame_time=-1, hold=False, quit=False, multi=False):
        if not self.visualization_enabled or frame_time == -1:
            for layer in self.layers:
                layer.push_forward(inputs)
                inputs = layer.outputs
            if not multi:
                self.outputs = layer.outputs[0]
            else:
                self.outputs = layer.outputs
            return

        if not self.vis.running:
            self.vis.create_screen()

        if frame_time > 0:
            self.vis.update_background()
            for idx, layer in enumerate(self.layers):
                layer.push_forward(inputs)
                self.vis.fill_neurons(idx, False)
                self.vis.show_activations(frame_time)
                inputs = layer.outputs
            self.vis.fill_neurons(idx, True)
            self.vis.show_activations(frame_time, last_frame=True, hold=hold, quit=quit)
            self.outputs = layer.outputs

    def random_sum(self, layer_sizes, n):
        rand_numbers = [0]*len(layer_sizes)
        usable_indexes = list(range(len(layer_sizes)))
        while sum(rand_numbers) < n:
            idx = random.choice(usable_indexes)
            if not rand_numbers[idx] < layer_sizes[idx]:
                usable_indexes.remove(idx); continue
            rand_numbers[idx] += 1 
        return rand_numbers

    def mutate(self, rate=1, scale=1, layers=ALL):
        if layers == self.ALL:
            layer_sizes = [layer.neurons for layer in self.layers]
            n = math.ceil(sum(layer_sizes)*scale)
            random_counts = self.random_sum(layer_sizes, n)
            for n, layer in zip(random_counts, self.layers):
                layer.mutate(rate, n)
        if layers == self.RANDOM:
            layer = random.choice(self.layers)
            n = math.ceil(layer.neurons * scale)
            layer.mutate(rate, n)
    
    def expand_verticaly(self, layer, size):
        self.layers[layer].biases = np.array([np.append(self.layers[layer].biases, np.zeros((size)))])
        self.layers[layer].neurons += size
        self.layers[layer].weights = np.vstack([self.layers[layer].weights.T, np.zeros((size, self.layers[layer].inputs))]).T
        self.layers[layer+1].weights = np.vstack([self.layers[layer+1].weights, np.zeros_like(self.layers[layer+1].weights[0])])
        self.layers[layer+1].inputs += 1
    
    def expand_horizontaly(self, index):
        new_layer = Layer(self.layers[index].neurons, self.layers[index].neurons, activation=Layer.SHARP_SIGMOID)
        new_layer.weights = np.zeros_like(new_layer.weights)
        np.fill_diagonal(new_layer.weights, 1)
        self.layers.insert(index+1, new_layer)

    def crossEntropyLoss(self, guess, target):
        guess = guess.clip(1e-99, 1)
        sum = -np.sum(np.log(guess) * target, axis=1)
        return np.mean(sum)
    
    def rootMeanSquareError(self, guess, target, multi=False):
        if not multi:
            power = np.power(guess - target, 2)
            sum = np.sum(power)
            return math.sqrt(sum)
        else:
            power = np.power(guess - target, 2)
            sum = np.sum(power, axis=1, keepdims=True)
            divided = np.divide(sum, len(guess[0]))
            return np.mean(np.sqrt(divided))
    
    def save(self, file_name='model.npy'):
        path = Path(f'{Path.cwd()}\models')
        if Path(f'{path}\{file_name}').exists():
            print('ERROR WHILE SAVING MODEL\n')
            print(f'File: "{path}\{file_name}" already exists!\n')
            return
        path.mkdir(exist_ok=True)
        path = Path(f'{path}\{file_name}')
        np.save(path, np.array(self.layers), allow_pickle=True)

    def load(self, file_name='model.npy'):
        path = Path(f'{Path.cwd()}\models\{file_name}')
        if not path.exists():
            print('ERROR WHILE LOADING MODEL\n')
            print(f'Path: "{path}" doesn\'t exists!\n')
            return
        self.layers = list(np.load(path, allow_pickle=True))


class Agent(NeuralNetwork):

    def __init__(self, layers):
        super().__init__(layers)
        
        self.idn = id(self)
        self.fitness = 0
        self.breeding_prob = 0

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
    NONE = 'none'
    RANDOM = 'random'

    def __init__(self, agent=None, size=None, unique=True, file_name=None):

        self.generations = 0

        self.include_parents = True
        self.mutation_rate = 0.5
        self.mutation_scale = 0.2
        self.pool_size = 4
        self.selection_method = self.TOP_X
        self.mutated_layers = NeuralNetwork.RANDOM
        self.expand = False

        self.vertical_freq = 20
        self.horizontal_freq = 100
        self.vertical_chance = 0.5
        self.horizontal_chance = 0.2

        self.n_layers = 1

        if file_name is not None:
            path = Path(f'{Path.cwd()}\populations\{file_name}')
            if not path.exists():
                print('ERROR WHILE LOADING POPULATION\n')
                print(f'Path: "{path}" doesn\'t exists!\n')
                return
            self.agents = list(np.load(path, allow_pickle=True))
            self.size = len(self.agents)
            return

        self.size = size

        self.agents = []
        for _ in range(size):
            new_agent = agent.copy()
            if unique:
                new_agent.mutate()
            new_agent.idn = id(new_agent)
            self.agents.append(new_agent)

    def fitness_func(self, fitness):
        return math.pow(fitness, 2)

    def calc_breeding_prob(self):
        fitness_sum = 0
        for agent in self.agents:
            fitness_sum += self.fitness_func(agent.fitness)
        if fitness_sum == 0:
            for agent in self.agents:
                agent.survival_prob = 1/self.size
            return
        for agent in self.agents:
            agent.breeding_prob = self.fitness_func(agent.fitness) / fitness_sum

    def create_mating_pool(self, size, method=None):
        if method is None:
            method = self.selection_method
        self.calc_breeding_prob()

        if method == self.RULETTE_WHEEL:
            mating_pool = []
            for _ in range(size):
                idx = 0
                rand = random.random()
                while rand > 0:
                    rand -= self.agents[idx].breeding_prob
                    idx += 1
                idx -= 1
                picked = self.agents[idx]
                mating_pool.append(picked)
            return mating_pool
        
        if method == self.TOP_X:
            mating_pool = sorted(self.agents, key=lambda a: a.breeding_prob, reverse=True)[:size]
            return mating_pool
    
    def expansion_settings(self, vertical_freq=None, horizontal_freq=None, vertical_chance=None, horizontal_chance=None):
        self.expand = vertical_freq or horizontal_freq or vertical_chance or horizontal_chance is not None
        if vertical_freq is not None:
            self.vertical_freq = vertical_freq
        if horizontal_freq is not None:
            self.horizontal_freq = horizontal_freq
        if vertical_chance is not None:
            self.vertical_chance = vertical_chance
        if horizontal_chance is not None:
            self.horizontal_chance = horizontal_chance

    def mutation_settings(self, mutation_rate=None, mutation_scale=None, mutated_layers=None, expand=None):
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
        if mutation_scale is not None:
            self.mutation_scale = mutation_scale
        if mutated_layers is not None:
            self.mutated_layers = mutated_layers
    
    def evolution_settings(self, include_parents=None, pool_size=None, selection_method=None, population_size=None):
        if include_parents is not None:
            self.include_parents = include_parents
        if pool_size is not None:
            self.pool_size = pool_size
        if selection_method is not None:
            self.selection_method = selection_method
        if population_size is not None:
            self.size = population_size
        
    
    def breeding_settings(self, n_layers=None):
        if n_layers is not None:
            self.n_layers = n_layers
        
    def cross(self, parent_1, parent_2, cross_size=None, layer_idx=None):

        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        if layer_idx is None: layer_idx = random.randint(0, len(parent_1.layers)-1)

        child_1_biases = child_1.layers[layer_idx].biases[0]
        child_2_biases = child_2.layers[layer_idx].biases[0]
        child_1_weights = child_1.layers[layer_idx].weights
        child_2_weights = child_2.layers[layer_idx].weights

        layer_size = len(child_1_biases)

        if cross_size is None: cross_size = int(layer_size / 2)

        split_point = 0 if cross_size == layer_size else random.randint(0, layer_size-cross_size-1)
       
        temp = np.copy(child_1_weights.T[split_point:split_point+cross_size])
        child_1_weights.T[split_point:split_point+cross_size] = np.copy(child_2_weights.T[split_point:split_point+cross_size])
        child_2_weights.T[split_point:split_point+cross_size] = temp
        temp = np.copy(child_1_biases[split_point:split_point+cross_size])
        child_1_biases[split_point:split_point+cross_size] = np.copy(child_2_biases[split_point:split_point+cross_size])
        child_2_biases[split_point:split_point+cross_size] = temp
        return [child_1, child_2]

    def evolve(self):
        self.calc_breeding_prob()

        new_agents = []

        mating_pool = self.create_mating_pool(self.pool_size, self.selection_method)

        if self.include_parents:
            for agent in mating_pool:
                new_agents.append(agent)

        while len(new_agents) < self.size:
            parents = random.sample(mating_pool, 2)
        
            offspring = self.cross(parents[0], parents[1])
            for child in offspring:
                child.mutate(rate=self.mutation_rate, scale=self.mutation_scale)

            new_agents.append(offspring[0])
            if len(new_agents) == self.size:
                break
            new_agents.append(offspring[1])

        ve, he = False, False
        if self.expand:
            if self.generations % self.vertical_freq == 0 and random.random() <= self.horizontal_chance:
                if len(self.agents[0].layers) == 1: ve = False
                elif len(self.agents[0].layers) == 2: ve = 0
                else: ve = random.randint(0, len(self.agents[0].layers)-2)
            if self.generations % self.horizontal_freq == 0 and random.random() <= self.vertical_chance:
                if len(self.agents[0].layers) == 1: he = False
                else: he = random.randint(0, len(self.agents[0].layers)-2)

        if ve is not False or he is not False:
            for agent in new_agents:
                if ve is not False: agent.expand_verticaly(ve, 1)
                if he is not False: agent.expand_horizontaly(he)

        self.agents = new_agents
        self.generations += 1
    
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
        np.save(path, np.array(self.agents), allow_pickle=True)

        