import numpy
import scipy.special #.expit - > сигмода
import matplotlib.pyplot as plt
import os, re
from PIL import Image, ImageOps    
import json
import itertools
from random import shuffle
import pandas as pd

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #матрицы весовых коэффициентов связей wih и who.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        #коэффициент обучения 
        self.lr = learningrate
        
        #функция активации сигмоида 
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    
    #тренеровка нейронной сети
    def train(self, inputs_list, targets_list):
        #преобразовать список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        
        #рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)                 # X = W*I-входящие сигналы      
        #рассчитать исхоящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)    #O=сигмоида(X-скрытый)
        
              
        #рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)          # X = W*I = W*O-скрытый
        #рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)      #O=сигмоида(X-выходящий)
        
        
        #ошибки выходного слоя = (целевое значение - фактическое значение)
        output_errors = targets - final_outputs
        
        #    ошибки скрытого слоя - это ошибки output_errors, 
        #    распределенные пропорционально весовым коэффициента связей 
        #    и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        
        #обновить весовые коэффициенты для связей между скрытым и выходным слоем
        self.who += abs(self.lr) * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        
        #обновить весовые коэффициенты для связей между входным и скрытым слоем
        self.wih += abs(self.lr) * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    #опрос нейронной сети
    def query(self, inputs_list, wih, who):
        #преобразовать список входных значений
        #в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        #рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(wih, inputs)                 # X = W*I-входящие сигналы      
        #рассчитать исхоящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)    #O=сигмоида(X-скрытый)
        
              
        #рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(who, hidden_outputs)          # X = W*I = W*O-скрытый
        #рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)      #O=сигмоида(X-выходящий)
        
        
        return final_outputs

#ТРЕНИРОВКА НЕЙРОННОЙ СЕТИ



class Training(neuralNetwork):
    pass