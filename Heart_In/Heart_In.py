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




#нормализовать данные в пределах (-1,1) пиковое значение будет 1
def normalized(array):  
    return np.round(-(array / CONST_int32 * 0.99), decimals = 3)

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

    
#генерирует массивы из сокращений указанной длины 
def systole_generator(array:"list" = [], scale_values:"int" = 1, size:"int" = 0) -> "вернет одномерный массив сокращений":
    cardiogram = []
    synthetic  = []
    for n,i in enumerate(array):
        scale   = scale_values[n]
        systole = array[n]
        systole_scaled = zoom(systole, scale).tolist()
        cardiogram.append(systole_scaled)
      
    for i in range(size):
        scale   = float(*random.choices(scale_values))
        systole = np.mean(np.array([ random.choices(array), random.choices(array) ]), axis=0 ).tolist()
        systole_scaled = zoom(systole, scale).tolist()
        synthetic+= systole_scaled

    return synthetic

def animation(name:"\\..." = "video.mp4", fps:"int" = 15):
    import matplotlib.animation as manimation

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()
    l, = plt.plot([], [], 'k-o')

    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    x0, y0 = 0, 0

    with writer.saving(fig, name, 100):
        for i in range(100):
            x0 += 0.1 * np.random.randn()
            y0 += 0.1 * np.random.randn()
            l.set_data(x0, y0)
            writer.grab_frame()

#визуализация массива или нескольких массивов
def render(*array:"lists of array", frequency:'Hz'= 1, colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
    #возвращает цвета из массива colors, когда кончатся стандартные цвета, будет генерировать случайные
    def get_color():
        for color in colors:
            yield color
    #экземпляр класса get_color
    color_buffer = get_color()
    
    for n, y in enumerate(array):
        #использует базовые цвета в наличии(colors), если заканчиваются, генерирует случайные
        if n < len(colors):
            color = next(color_buffer)        
        else:
            color = (np.random.randint(255, size=(1, 3))/255).flat
            
        x  = np.linspace(0, frequency*len(y), len(y), endpoint=False)
        plt.plot(x, y, color=color, marker ='')

    plt.show()

#сохранение массива
def save(array:"list" = [], name:"name.json" = "cardiogram.json"):
    with open(name, 'w') as cardiogram:
        json.dump(buffer, cardiogram)


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

#загрузить масси кардиограммы
path = r"C:\Users\o.zaitsev\Source\Repos\neuralNetwork\Heart-In\0a0ab63fe6bbf7ec785c62eef3c6d654.jpg"
#константное число, тип чисел в массиве, надо знать для конвертации, так как он в бинарный
CONST_int32 = 2147483647
#считать с файла в numpy массив
array = np.fromfile(path, dtype='i4', count=-1, sep='')[15000:50000]
#подменить "Nan" ноль
array[np.isnan(array)] = 0   

#нормализовать значения в пределах [-1,1]
y = normalized(array)

#удалить значения шума
y = y[y!=0.0]
y = y[y!=-0.078]

# найти R пики с позициями по частоте array[0] - позиции, array[1] - значения R пиков 
R_peak = peakValues(y, step = 190,  frequency = 1)

#разбить массив на куски по сокращениям используя данный от "peakValues" - array[0]  
systoles = systole_separator(R_peak[0], y)



buffer = systole_generator(array = systoles[0], scale_values = systoles[1], size = 1000)

save(buffer)

#render(y, R_peak[1])

#render(sum(buffer, []))