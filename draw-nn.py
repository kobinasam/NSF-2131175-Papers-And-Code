
from math import cos, sin, atan
import os

from matplotlib import pyplot
import numpy as np

class Neuron():
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        circle.set_color(self.color)
        pyplot.gca().add_patch(circle)


class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, color_map):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.color_map = color_map
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            color = self.color_map[iteration, -1] if self.color_map is not None else 'k'
            neuron = Neuron(x, self.y, color)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, color):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
        line.set_color(color)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron_i, neuron in enumerate(self.neurons):
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_neuron_i, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                    color = self.color_map[neuron_i, previous_neuron_i]
                    self.__line_between_two_neurons(neuron, previous_layer_neuron, color)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, self.y, 'Input Layer', fontsize = 12)
        elif layerType == -1:
            pyplot.text(x_text, self.y, 'Output Layer', fontsize = 12)
        else:
            pyplot.text(x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 12)

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, color_map):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, color_map)
        self.layers.append(layer)

    def draw(self):
        pyplot.figure()
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            layer.draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()

class DrawNN():
    def __init__( self, neural_network, color_maps):
        self.neural_network = neural_network
        self.color_maps = color_maps

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for l, cm in zip(self.neural_network, [None] + self.color_maps):
            network.add_layer(l, cm)
        network.draw()

def create_color_map(weight_matrix):
    color_map = np.full(weight_matrix.shape, 'k')
    color_map[weight_matrix == 0] = 'r'
    return color_map

if __name__ == "__main__":

    # TO SEE MAXWELL's RESULTS
    os.chdir(os.path.join(os.getcwd(), 'main', 'lmbp', 'lmbp', 'untitled folder'))    
    weights_dropped=30

    # TO SEE KUSHAL'S RESULTS
    # os.chdir(os.path.join(os.getcwd(), 'main', 'lmbp', 'lmbp', 'kushal_results'))    
    # weights_dropped=50

    # If filenames change, then update this
    filenames = [f'w{i}_best_{weights_dropped}_dropped.csv' for i in range(1, 4)]

    weights = [np.genfromtxt(fn, delimiter=',') for fn in filenames]
    color_maps = [create_color_map(w) for w in weights]

    drawer = DrawNN([4, 6, 6, 2], color_maps=color_maps)
    drawer.draw()
