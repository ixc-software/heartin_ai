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


#найти пики
def maxValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1024 ) -> "вернет пики с позициями(идексами)":
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
        y_max = max(array[i:i+step])
        x_position = i + list(array[i:i+step]).index(y_max)
        
        
        values_x += [x[x_position]]
        values_y += [y_max]
    
    values_x +=[frequency*len(array)]
    values_y +=[0]
    return values_x, values_y

def minValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1024 ) -> "вернет пики с позициями(идексами)":
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
        y_max = min(array[i:i+step])
        x_position = i + list(array[i:i+step]).index(y_max)

        values_x += [x[x_position]]
        values_y += [y_max]
    
    values_x +=[frequency*len(array)]
    values_y +=[0]
    return values_x, values_y


#создать массив с шагом 1024Hz
x  = np.linspace(0, 1024*y.size, y.size,endpoint=False)

# найти R пики с позициями по частоте
R_max = maxValues(y, step = 5)
R_min = minValues(y, step = 5)


#склейка по глубине
"""
x = np.array([5,4,3,2,1])
y = np.array([4,3,2,1,0])
[[5 4]
 [4 3]
 [3 2]
 [2 1]
 [1 0]]
 """
X_train = np.dstack((R_max[1], R_min[1])).reshape(-1,2)


#создать таблицу pandas:
"""
            0      1
_____________________
0         0.0 -0.004
_____________________
1      1024.0  0.008
_____________________
...
"""
frequency_dataframe = pd.DataFrame(X_train)


#создать матрицу рассеяния
#grr =  pd.plotting.scatter_matrix(frequency_dataframe, color=['#e41a1c', '#377eb8'], figsize=(10,10), s=100, alpha=0.8)


print(len(R_max[0]),len(R_max[1]))
print(len(R_min[0]),len(R_min[1]))

plt.plot(R_max[0],R_max[1], color='r', marker ='*')
plt.plot(R_min[0],R_min[1], color='g', marker ='*')
plt.plot(x,y, color='b')
plt.show()