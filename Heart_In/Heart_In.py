import numpy as np
from math import factorial
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
plt.rc('font', family='Verdana')
from scipy.signal import argrelextrema
from scipy.ndimage import zoom #делает скейл массива [0,1] -> [0, 0.5, 1] <- [0,1]
import random
from itertools import combinations
import json

path = r"C:\Users\oleks\Source\Repos\heartin_ai\Heart_In\0a0ab63fe6bbf7ec785c62eef3c6d654.jpg"
CONST_int32 = 2147483647

#нормализовать данные в пределах (-1,1) пиковое значение будет 1
def normalized(array):  
    return np.round(-(array / CONST_int32 * 0.99), decimals = 3)

#считать с файла в numpy массив
array = np.fromfile(path, dtype='i4', count=-1, sep='')[15000:50000]

#подменить "Nan" ноль
array[np.isnan(array)] = 0   

#нормализовать значения в пределах [-1,1]
y = normalized(array)

#удалить значения шума
y = y[y!=0.0]
y = y[y!=-0.078]

#найти пики +
def maxValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1 ) -> "вернет пики с позициями(индексами)":
    x  = np.linspace(0, frequency*len(array), len(array), endpoint=False)
    
    values_x = [0]
    values_y = [0]

    for i in range(0, len(array), step):
        buffer = list(array[i:i+step])
        y_max = max(buffer)
        x_position = i + buffer.index(y_max)
        
        
        values_x += [x[x_position]]
        values_y += [y_max]
    
    values_x +=[frequency*len(array)]
    values_y +=[0]
    return values_x, values_y

#найти пики -
def minValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1 ) -> "вернет пики с позициями(идексами)":
    x  = np.linspace(0, frequency*len(array), len(array), endpoint=False)
    
    values_x = [0]
    values_y = [0]

    for i in range(0, len(array), step):
        buffer = list(array[i:i+step])
        y_min = min(buffer)
        x_position = i + buffer.index(y_min)

        values_x += [x[x_position]]
        values_y += [y_min]
    
    values_x +=[frequency*len(array)]
    values_y +=[0]
    return values_x, values_y


#найти пики с условием поиска от начала пиков, возвращает пики, а так же разметку values_x, values_y
def peakValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1) -> "вернет пики с позициями(идексами)":
    x  = np.linspace(0, frequency*len(array), len(array), endpoint=False)
     
    index = 0
    values_x = []
    values_y = []
    
    while True:
        if len(array[index:index+step]) < step//5:
            break
        buffer = list(array[index:index+step])
        #находим максимум на отрезке
        y_max = max(buffer)
        #находим позицию максимума на отрезке
        x_position = index + buffer.index(y_max)
        #записываем максимум
        values_y += [y_max]
        #записываем позицию максимума
        values_x += [x[x_position]]
        #делаем отступ от стартового значения в половину шага
        index = x_position + step//2
        
    return values_x, values_y


#создает список, где на каждой позиции массив с одним сокращением, используй данные от "peakValues" - array[0]
def systole_separator(position:"[position]"= [], data:"[data]"= [], size:'const' = 200) -> "вернет список, на каждой позиции сокращение":
    systoles = []
    scale_x  = []
    try:
        for n,i in enumerate(position):

            startIndex = int(position[n])
            endIndex   = int(position[n + 1])
            
            buffer = list(data[startIndex:endIndex])
            #считаем коэффициент скейла в процентах, на 100 не умножаем что бы получить коэффициент для умножения
            buffer_scale = (size/len(buffer))
            #втискиваем массив в 200 позиций
            buffer = zoom(buffer, buffer_scale)
            #добавляем сокращение в общий массив
            systoles.append(buffer)
            #добавляем коэффициент скейла сокращения
            scale_x.append(buffer_scale)
            
    except IndexError:
        print()
        
    finally:
        return systoles, scale_x

#генерирует одномерный массив из сокращений указанной длины 
def systole_generator(array:"list" = [], scale:"int" = 1, size:"int" = 0) -> "вернет одномерный массив сокращений":
    cardiogram = []
    synthetic  = []
    for n,i in enumerate(array):
        buffer = list(zoom(array[n], scale[n]))
        cardiogram.append(buffer)
    
    for i in range(size):
        synthetic += random.choices(cardiogram)

    return sum(synthetic, [])

#визуализация массива
def render(y:"list"= [], frequency:'Hz'= 1, color='b'):
    x  = np.linspace(0, frequency*len(y), len(y), endpoint=False)
    plt.plot(x, y, color=color, marker ='')

#сохранение массива
def save(array:"list" = [], name:"name.json" = "cardiogram.json"):
    with open(name, 'w') as cardiogram:
        json.dump(buffer, cardiogram)


#создать массив с шагом 1Hz
x  = np.linspace(0, 1*y.size, y.size,endpoint=False)

# найти R пики с позициями по частоте array[0] - позиции, array[1] - значения R пиков 
R_peak = peakValues(y, step = 190,  frequency = 1)

#разбить массив на куски по сокращениям используя данный от "peakValues" - array[0]  
systoles = systole_separator(R_peak[0], y)


#склейка по глубине
#X_train = np.dstack((R_max[1], R_min[1])).reshape(-1,2)
"""
x = np.array([5,4,3,2,1])
y = np.array([4,3,2,1,0])
[[5 4]
 [4 3]
 [3 2]
 [2 1]
 [1 0]]
 """


#создать таблицу pandas:
#frequency_dataframe = pd.DataFrame(X_train)
"""
            0      1
_____________________
0         0.0 -0.004
_____________________
1      1024.0  0.008
_____________________

"""

#создать матрицу рассеяния
#grr =  pd.plotting.scatter_matrix(frequency_dataframe, color=['#e41a1c', '#377eb8'], figsize=(10,10), s=100, alpha=0.8)


buffer = systole_generator(array = systoles[0], scale = systoles[1], size = 1000)
save(buffer)
render(buffer)


plt.show()