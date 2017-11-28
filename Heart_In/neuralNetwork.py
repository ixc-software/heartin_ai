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
    
    #тренировка нейронной сети
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
    
    def __init__(self):
        #запустить базовый конструктор сети 
        neuralNetwork.__init__(self, 28*28, 50, 10, 0.3)

        #коэффициент обучения 
        self.lr = 0.3

        #задать количество узлов во входном, скрытом и выходном слое
        self.inodes = 28*28
        self.hnodes = 500
        self.onodes = 10
        
        #матрицы весовых коэффициентов связей wih и who.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        

    #подготовка любой картнки, конвертация ее в массив numpy с размером 28х28 
    def image_ready(self, path:"...\\image.PNG" = "String", type:" RGB or L" = "L", size:"pixel typle" =(28,28)):
        """"will return data in format network needs"""

        image = Image.open(path).convert(type).resize(size)
        inverted_image = ImageOps.invert(image)
        matplotlib.pyplot.imshow(inverted_image, cmap='Greys', interpolation = 'None')

        #маштабировать и сместить входные значения в пределах от 0.01 - 1.00  (любое значение в пределах 255 / 255.0 * 0.99) + 0.01)
        return ((numpy.asfarray(inverted_image.getdata())/ 255.0 * 0.99) + 0.01) # -> asfarray конвертирует массив в float


    #перебрать все записи в тренировочном наборе
    def training(self, path=r"C:\Users\o.zaitsev\Source\Repos\neuralNetwork\neuralNetwork\numbers\\"):
        #начать обучение с стартовыми весами уже полученными ранее
        
        wih_file =  open("wih.json", 'r')
        who_file =  open("who.json", 'r')

        self.wih = numpy.array(json.load(wih_file))
        self.who = numpy.array(json.load(who_file))


        images = os.listdir(path)
        shuffle(images)
        for image in images:
        
            fullpath = path + image 
            inputs = self.image_ready(fullpath)
        
            #создать целевые выходные значения(все равны 0,01 за исключением желаемого маркерного значения 0,99)
            targets = numpy.zeros(self.onodes) + 0.01
        
            #целевое маркерное значение для данной записи
            targets[int(re.findall("(.*)_.*", image)[0])] = 0.99
            self.train(inputs, targets)

    #ПРОВЕРКА НЕЙРОННОЙ СЕТИ И ДООБУЧЕНИЕ
    def percentage_of_correct_answers(self, path:"...\\examine\\" = "String",use_weights = True) ->"вернет процент правильных ответов":
        """надо указать путь к папке где картинки 
       руками, для провеки процента правильных ответов"""
    
        wih_file =  open("wih.json", 'r')
        who_file =  open("who.json", 'r')
        wih = numpy.array(json.load(wih_file))
        who = numpy.array(json.load(who_file))

        images = os.listdir(path)
        number_of_pictures = len(images)
        guessed = 0

        for image in images:
            fullpath = path + image 
            arr = self.query(self.image_ready(fullpath), wih, who)
            arr_max = numpy.amax(arr)
            
            number = int(re.findall("(.*)_.*", image)[0])
            
            if numpy.where(arr==arr_max)[0] == number:
                use_weights = True
                guessed += 1 
            else:
                use_weights = False
        
        #считает процент правильных ответов
        correct_answers = py(guessed *100)/number_of_pictures
        
        wih_file.close() 
        who_file.close()
        return correct_answers

    def save_weights(self):
        try:
            with open("wih.json", 'w') as wih:
                json.dump(self.wih.tolist(), wih)

        except ValueError:
            pass

        try:
            with open("who.json", 'w') as who:
                json.dump(self.who.tolist(), who)
        except ValueError:
            pass

    #проверка исходных весов, если распознала больше чем перед этим, использем веса
    def examine(self, path:"...\\examine\\" = "String"):  
        
        #сохранить базовые случайные веса
        self.save_weights()

        while True:
            if int(self.percentage_of_correct_answers(path)) > 70:
                break
            else:
                self.save_weights()

            #первая тренировка на базе случайных весов
            self.training(r"C:\Users\o.zaitsev\Source\Repos\neuralNetwork\neuralNetwork\numbers\\")
            print("\tpercentage_of_correct_answers  =", self.percentage_of_correct_answers(path), "\tlr = ", self.lr)
           


        old_correct_answers = self.percentage_of_correct_answers(path)
        while True:
            #тренировка на базе обновленных весов(первый раз всегда будет лучше, чем случайные
            self.training(r"C:\Users\o.zaitsev\Source\Repos\neuralNetwork\neuralNetwork\numbers\\")
            new_correct_answers = self.percentage_of_correct_answers(path)



            if int(self.percentage_of_correct_answers(path)) > 80:
                break
            #если распознала больше чем перед этим, записываем обновленные веса в файл и уменьшаем шаг
            if  new_correct_answers >= old_correct_answers:
                self.save_weights()
                self.lr -= 0.010
                old_correct_answers = self.percentage_of_correct_answers(path)


            print("old_correct_answers = ", old_correct_answers, "\tpercentage_of_correct_answers  =", new_correct_answers, "\tlr = ", self.lr)
        print("FINAL RESULT =", new_correct_answers)


#examine_images = r"C:\Users\o.zaitsev\Source\Repos\neuralNetwork\neuralNetwork\examine\\"
run = Training()
#run.examine(examine_images)