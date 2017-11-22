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
array = np.fromfile(path, dtype='i4', count=-1, sep='')[:2500]

#подменить "Nan" ноль
array[np.isnan(array)] = 0   

#нормализовать значения в пределах [-1,1]
y = normalized(array)

#удалить значения шума
y = y[y!=0.0]
y = y[y!=-0.078]


#найти пики
maxInd = argrelextrema(y, np.greater)
R = y[maxInd]  # array([5, 3, 6])

#создать массив с шагом 1024Hz
x  = np.linspace(0, 1024*y.size, y.size,endpoint=False)

# X массив длиной в количество элементов R
xR = np.linspace(0, 1024*y.size, R.size,endpoint=False)


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
X_train = np.dstack((x, y)).reshape(-1,2)


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



plt.plot(xR,R, color='r', marker ='*')
plt.plot(x,y, color='b')
plt.show()
