from os import access
from matplotlib.pyplot import show
import numpy as np
from math import ceil
import cv2 as cv
import pygame
import time
from keyboard import is_pressed


class Layer:
    
    # activation functions
    SIGMOID = 1
    RELU = 2
    SOFTMAX = 3

    # normalization functions
    NONE = 4
    MIN_MAX = 5
    DIV_BY_MAX = 6

    # mutation history
    last_weights_mutation = None
    last_biases_mutation = None


    def __init__(self, inputs, neurons, activation=SIGMOID, normalization=NONE, rand_weights=True, rand_biases=True):
        # creates random array with size number of inputs times number of neurons
        # or creates array with only ones with size number of inputs times number of neurons
        if rand_weights: self.weights = np.random.uniform(-rand_weights, rand_weights, (inputs, neurons))
        else: self.weights = np.ones((inputs, neurons))

        self.neurons = neurons
        self.inputs = inputs

        # creates random array with size number of neurons
        # or creates array with only ones with size number of neurons
        if rand_biases: self.biases = np.random.uniform(-rand_biases, rand_biases, (1, neurons))
        else: self.biases = np.zeros((1, neurons))

        self.activation = activation
        self.normalization = normalization

    
    def push_forward(self, inputs):
        inputs = np.array(inputs)
        if 'array' not in str(type(inputs[0])):
            inputs = inputs.reshape((1, -1))
        self.input_values = inputs

        # normalizes inputs with specified method
        match self.normalization:
            case self.MIN_MAX:
                pass
            case self.DIV_BY_MAX:
                inputs = inputs.clip(1e-99, 1)
                maximum = np.max(inputs, axis=1, keepdims=True)
                inputs = np.divide(inputs, maximum)
            case self.NONE:
                pass
                
        # does dot product on inputs and weights for every sample
        neuron_sums = np.dot(inputs, self.weights)

        # adds biases to neurons outputs for every sample
        outputs = neuron_sums + self.biases

        # aplies specified activation funtion
        match self.activation:
            case self.SIGMOID:
                self.outputs = np.divide(1, 1+np.exp(-np.clip(outputs, -15, 15)))
            case self.RELU:
                self.outputs = np.maximum(0, outputs)
            case self.SOFTMAX:
                inputs = inputs.clip(1e-99, 1)
                e = np.exp(outputs)
                e = np.divide(e, np.max(e, axis=1, keepdims=True))
                self.outputs = np.divide(e, np.sum(e, axis=1, keepdims=True))
        

    def mutate(self, rate, repeat=False, scale=1):

        # creates random array with size of weights and biases arrays if reapeat is false
        # if repeat is true, it uses same random array as last time
        if repeat and self.last_weights_mutation is not None:
            weights_mutation = self.last_weights_mutation
            biases_mutation = self.last_biases_mutation
        
        else:
            weights_mutation = np.random.uniform(-rate, rate, self.weights.shape)
            biases_mutation = np.random.uniform(-rate, rate, self.biases.shape)

            # leavs only scales * 100 % mutations in
            weights_mutation_scale = np.zeros_like(weights_mutation)
            weights_mutation_scale = weights_mutation_scale.reshape((1, -1))
            weights_mutation_scale[:,:ceil(weights_mutation_scale.shape[1] * scale)] = 1
            np.random.shuffle(weights_mutation_scale[0])
            weights_mutation_scale = np.reshape(weights_mutation_scale, weights_mutation.shape)
            weights_mutation *= weights_mutation_scale

            biases_mutation_scale = np.zeros_like(biases_mutation)
            biases_mutation_scale[:,:ceil(biases_mutation_scale.shape[1] * scale)] = 1
            np.random.shuffle(biases_mutation_scale[0])
            biases_mutation_scale = np.reshape(biases_mutation_scale, biases_mutation.shape)
            biases_mutation *= biases_mutation_scale

        # saves current mutation for easy repetion
        self.last_weights_mutation = weights_mutation
        self.last_biases_mutation = biases_mutation

        # aplies mutation to weights and biases
        self.weights += weights_mutation
        self.biases += biases_mutation
    
    def reverse_mutation(self):
        self.weights -= self.last_weights_mutation
        self.biases -= self.last_biases_mutation


class NeuralNetwork:

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

        AUTO = 1

        def __init__(self, layers, screen_size=(1600, 800), left_layer_ofset=0.1, window_name='Neural Network', neuron_size=AUTO):
            self.window_name = window_name
            self.layers = layers
            self.architecture = np.array([[l.inputs, l.neurons] for l in layers], dtype=object)
            self.width = screen_size[0]
            self.height = screen_size[1]
            self.screen_size = screen_size
            self.left_layer_ofset = int(left_layer_ofset*self.width)

            if neuron_size is self.AUTO:
                self.radius_neuron = int(self.height / (self.architecture.max()*2 + 2) / 2)
            else:
                self.radius_neuron = int(neuron_size)
            self.radius_neuron = max(1, self.radius_neuron)
            
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
                        bias = biases[idx]
                        color = self._get_bias_color(bias)
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

    def enable_visualization(self, size=(1600, 800), neuron_size=Visualization.AUTO):
        self.vis = self.Visualization(screen_size=size, layers=self.layers, neuron_size=neuron_size)
        self.visualization_enabled = True


    def push_forward(self, inputs, frame_time=-1, hold=False, quit=False):
        if not self.visualization_enabled or frame_time == -1:
            for layer in self.layers:
                layer.push_forward(inputs)
                inputs = layer.outputs
            self.outputs = layer.outputs

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


    
    def mutate(self, rate=1, repeat=False, scale=1):
        for layer in self.layers:
            layer.mutate(rate=rate, repeat=repeat, scale=scale)
    

    def reverse_mutation(self):
        for layer in self.layers:
            layer.reverse_mutation()
    

    def crossEntropyLoss(self, guess, target):
        guess = guess.clip(1e-99, 1)
        sum = -np.sum(np.log(guess) * target, axis=1)
        return np.mean(sum)
    
    
    def rootMeanSquareError(self, guess, target):
        power = np.power(guess - target, 2)
        sum = np.sum(power, axis=1, keepdims=True)
        divided = np.divide(sum, len(guess[0]))
        return np.mean(np.sqrt(divided))
    
    def save(self, file_name=None):
        if file_name is None:
            file_name = f'model.npy'
            np.save(file_name, np.array(self.layers), allow_pickle=True)
    
    def load(self, file_name=None):
        if file_name is None:
            file_name = f'model.npy'
            self.layers = list(np.load(file_name, allow_pickle=True))

if __name__ == '__main__':
    # create Layer objects
    l1 = Layer(4, 16, normalization=Layer.DIV_BY_MAX, rand_weights=False)
    l2 = Layer(16, 8, rand_biases=False)
    l3 = Layer(8, 4, activation=Layer.SOFTMAX)

    # create NeuralNetwork object from list of Layer objects
    net = NeuralNetwork([l1, l2, l3])
    # enables visualization
    net.enable_visualization()
    # animates building net
    net.vis.show_building_net()
    # show finnished net
    net.vis.show_net()

    for _ in range(200):
        # mutates all layers in network randomly by 0.002
        net.mutate(0.02)
        net.vis.show_net(frame_time=-1)
    
    net.vis.show_net()
    for _ in range(600):
        # mutates all layers in network same as last time
        net.mutate(0.02, repeat=True)
        net.vis.show_net(frame_time=-1)
    
    net.vis.show_net()

    # pushes data throught the network - if frame_time == -1: nothing is shown
    net.push_forward([0, 0, 0, 1], frame_time=0.5)
    net.push_forward([1, 0, 1, 0], frame_time=0.5)
    # hold = True holds frame on screen, quit will quit entire script after key is pressed
    net.push_forward([1, 1, 1, 1], frame_time=0.5, hold=True, quit=True)
