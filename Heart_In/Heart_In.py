import numpy as np
from math import factorial
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
plt.rc('font', family='Verdana')
from scipy.signal import argrelextrema

path = r"C:\Users\oleks\Source\Repos\neuralNetworkource\Heart-In\0a0ab63fe6bbf7ec785c62eef3c6d654.jpg"
CONST_int32 = 2147483647

#нормализовать данные в пределах (-1,1) пиковое значение будет 1
def normalized(array):  
    return np.round(-(array / CONST_int32 * 0.99), decimals = 3)

#считать с файла в numpy массив
array = np.fromfile(path, dtype='i4', count=-1, sep='')

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
    """  X Y 
       [[5 4]
        [4 3]
        [3 2]
        [2 1]
        [1 0]]
        """
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
    """  X Y 
       [[5 4]
        [4 3]
        [3 2]
        [2 1]
        [1 0]]
        """
    for i in range(0, len(array), step):
        buffer = list(array[i:i+step])
        y_min = min(buffer)
        x_position = i + buffer.index(y_min)

        values_x += [x[x_position]]
        values_y += [y_min]
    
    values_x +=[frequency*len(array)]
    values_y +=[0]
    return values_x, values_y


#найти пики с условием поиска от начала пиков 
def peakValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1024 ) -> "вернет пики с позициями(идексами)":
    x  = np.linspace(0, frequency*len(array), len(array), endpoint=False)
    
    index = 0
    values_x = [0]
    values_y = [0]
    """  X Y 
       [[5 4]
        [4 3]
        [3 2]
        [2 1]
        [1 0]]
        """
    
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
        

        index = x_position + step//2


    values_x +=[frequency*len(array)]
    values_y +=[0]
    return values_x, values_y




#создать массив с шагом 1Hz
x  = np.linspace(0, 1*y.size, y.size,endpoint=False)

# найти R пики с позициями по частоте
#R_max = maxValues(y, step = 175)
#R_min = minValues(y, step = 175)
R_peak = peakValues(y, step = 200,  frequency = 1)


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


plt.plot(R_peak[0],R_peak[1], color='r', marker ='*')
#plt.plot(R_max[0],R_max[1], color='r', marker ='*')
#plt.plot(R_min[0],R_min[1], color='g', marker ='*')
plt.plot(x,y, color='b')
plt.show()