import numpy as np
from math import factorial, sqrt
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import pandas as pd
from IPython.display import display
plt.rc('font', family='Verdana')
from scipy.signal import argrelextrema
from scipy.ndimage import zoom #делает скейл массива [0,1] -> [0, 0.5, 1] <- [0,1]
import random
from itertools import combinations
import json
import sys, os
import imageio
import time
import itertools
import requests
import threading
from multiprocessing import Pool, Process


def timer(func):
    def wrapper(*args):
        startTime = time.time()
        func(*args)
        endTime   = time.time()
        print("func - {}, lead time :{} sec.".format(func.__name__, endTime - startTime))
    return wrapper

#нормализовать данные в пределах (-1,1) пиковое значение будет 1
def normalized(array):
    #константное число, тип чисел в массиве
    CONST_int32 = 2147483647
    return np.round(-(array / CONST_int32 * 0.99), decimals = 3)


#найти пики +
def maxValues(array:"list"= [], step:"int"= 5, frequency:'Hz'= 1 ) -> "вернет пики с позициями(индексами)":
    print("function 'maxValues' in progress \n")
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
    print("function 'minValues' in progress \n")
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


#найти пики с условием поиска от начала пиков, возвращает пики, разметку values_x, values_y
def get_peaks(cardiogram:"list"= [], step:"int"= 185, frequency:'Hz'= 1) -> "вернет пики с позициями(идексами)":
    print("function 'get_peaks' in progress \n")
    x  = np.linspace(0, frequency*len(cardiogram), len(cardiogram), endpoint=False)
     
    index = 0
    values_x = []
    values_y = []
    
    while True:
        if len(cardiogram[index:index+step]) < step//5:
            break
        buffer = list(cardiogram[index:index+step])
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


#посчитать коэффициент корреляции Пирсона между двумя значениями
def pearson_correlation(first, second) ->"вернет от -1 до 1, 1 - значит 100% сходство":
    #количество элементов массива
    array_len = len(first)

    #вычислить сумму значений массива
    first_sum  = sum(first)
    second_sum = sum(second)
       
    #вычислить сумму квадратов массива
    first_pow_sum  = sum([pow(i, 2) for i in  first])
    second_pow_sum = sum([pow(i, 2) for i in second])

    #вычислить сумму произведений 
    first_second_product_sum = sum([i[0]*i[1] for i in zip(first, second)])

    #вычислить коэффициент Пирсона
    num = first_second_product_sum - (first_sum * second_sum/array_len)
    den = sqrt((first_pow_sum - pow(first_sum, 2)/array_len)*(second_pow_sum - pow(second_sum, 2)/array_len))

    if den == 0: return 0

    correlation = num/den
    return correlation


#визуализация массива или несколько массивов
def render(*array:"lists of array", frequency:'Hz'= 1, colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']):
    print("function 'render' in progress \n")
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


#создает GIF анимацию из массива
def animation(array:"list"=[], path:"\\..." = "image.png", fps:"int" = 15, max_time:"second" = 100):
    print("function 'animation' in progress \n")
    fig = plt.figure()
    x  = np.linspace(0, 1*len(array), len(array), endpoint=False)
    
    #узнаем время запуска
    start_time = time.time()
    try:
        for n, y in enumerate(array):
            #указать размер рамки по X
            plt.xlim(0+n, 512+n)
            #указать размер рамки по Y
            plt.ylim(-0.5, 0.5)
            #нарисовать массивы по (x, y) 
            plt.plot(x[n:512+n], array[n:512+n], color='r', marker ='')
        
            fig.savefig(path + "image_{0:010}.png".format(n))
            #узнаем текущие время 
            current_time = time.time()
            #если прошло больше секунд чем в переменной max_time, возбуждаем исключение
            if current_time - start_time > max_time:
                raise Exception("Time is over!")

    except Exception as Error:
        print(Error)

    finally:
        filenames  = [i for i in [path + i for i in os.listdir(path)] if i.endswith(".png")]

        with imageio.get_writer(r"{}\image.gif".format(path), mode='I', fps=fps) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for image in filenames:
            os.remove(image)
        
        #открывает папку с GIF
        os.startfile(path)

        
#загрузить json
def load_json(path:"\\..." = "cardiogram.json"):
    print("function 'load_json' in progress \n")
    with open(path) as json_data:
        return json.load(json_data)

    
#сохранение массива в формате json
def save_json(array:"list" = [], path:"name.json" = "cardiogram.json"):
    print("function 'save_json' in progress \n")
    with open(path, 'w') as cardiogram:
        json.dump(array, cardiogram)

        

#создает списки, где на каждой позиции массив с одним сокращением и его длиной, используй данные от "get_peaks" - array[0]
def separate_by_peak(offcuts:"list"= [], cardiogram:"list"= [], length:'const' = 200) -> "вернет словарь":
    print("function 'separate_by_peak' in progress \n")
    assert (type(length) == int), "(from separate_by_peak)size should be INT, but - {}".format(size)
    systoles = []
    lengths  = []
    for n,i in enumerate(offcuts[:-1]):

        startIndex = int(offcuts[n])
        endIndex   = int(offcuts[n + 1])
        buffer = list(cardiogram[startIndex:endIndex])
            
        #считаем коэффициент скейла в процентах, на 100 не умножаем что бы получить коэффициент для умножения
        buffer_lengths = (length/len(buffer))
        #втискиваем массив в 200 позиций
        buffer = zoom(buffer, buffer_lengths).tolist()
        #добавляем сокращение в общий массив
        systoles.append(buffer)
        #добавляем коэффициент скейла сокращения
        lengths.append(buffer_lengths)
    
    #возвращает словарь сокращений и их размеры [[сокращения,..][коэффициенты скейла,..]]
    return {"systoles":systoles, "lengths":lengths}


#скачивает с сервера определенное количество кардиограмм, определенных размеров
def downloader(url:"string"= r"https://", list_uploads:"links list" = str(), save_path:"string" = r"binary", from_size:"int" = 300000):
    print("function 'downloader' in progress \n")
    
    #функция скачивания по ссылке и хешу
    def threading_requests(links, names):
                response = requests.get(link, stream=True, verify=False)
                with open(os.path.join(save_path,name), 'wb') as f:
                    f.write(response.content)
                del response

    #подготавливает ссылки и хеш имена для функции скачивания 
    links  = []
    names  = []
    with open(list_uploads, "r") as f:
        for line in f:
            info = line.split()[-5:]
            data_size = info[0 ]
            data_hash = info[-1]
            if int(data_size) > from_size:
                links.append(os.path.join(url,data_hash))
                names.append(data_hash)
               
        for link, name in zip(links, names):
            print(link)
            #запуск потока на скачивание
            threading.Thread(target=lambda: threading_requests(links, names)).start()


#генерирует массивы из сокращений указанной длины 
def systoles_generator(cardiogram:"list" = [], lengths:"list" = [], size:"int" = 0) -> "вернет словарь":
    print("function 'systoles_generator' in progress \n")
    assert (type(size) == int), "(from systoles_generator)size should be INT, but - {}".format(size)

    original   = []
    synthetic  = []

    #возвращает массиву сокращений исходный скейл для каждого, приводит к оригинальному виду
    for n,i in enumerate(cardiogram[:size]):
        scale   = lengths[n]
        systole = [cardiogram[n]]
        #возвращает исходную длину для сокращения
        systole_scaled = zoom(systole, scale).tolist()
        #фильтрует неправильно обрезанные пики, потенциально может попасть, правильная аномалия(например жел. экстрасистола)
        if  max(systole_scaled[0][-5:-1]) > systole_scaled[0][-1]: 
            systole_scaled = [systole_scaled[0][:-5]]
            #render(*systole_scaled)
            
        if max(systole_scaled[0][1:5]) > systole_scaled[0][0]:
            systole_scaled = [systole_scaled[0][5:]]
            #render(*systole_scaled)
            
        original+= systole_scaled

    
    
    #создает синтетический набор сокращений скрещенный случайным образом друг с другом
    for i in range(size): 
        #взять случайный коэффициент скейла для сокращения из общего массива
        scale   = float(*random.choices(lengths))
        #получить сокращение созданное из двух случайно выбранных, на выходе среднее значение между ними 
        systole = np.mean(np.array([ random.choices(cardiogram), random.choices(cardiogram) ]), axis=0 ).tolist()
        
        #возвращает исходную длину для сокращения
        systole_scaled = zoom(systole, scale).tolist()
        
        #фильтрует неправильно обрезанные пики, потенциально может попасть, правильная аномалия(например жел. экстрасистола)
        if  max(systole_scaled[0][-5:-1]) > systole_scaled[0][-1]: 
            systole_scaled = [systole_scaled[0][:-5]]
            #render(*systole_scaled)
            
        if max(systole_scaled[0][1:5]) > systole_scaled[0][0]:
            systole_scaled = [systole_scaled[0][5:]]
            #render(*systole_scaled)
            
        synthetic+= systole_scaled

    return {"synthetic":synthetic, "original":original}


#соединяет все кардиограммы из папки в один массив
def merge_cardiograms(path:"string" = r"binary", amount:"int" = 0)->"":
    print("function 'put_together_cardiograms' in progress: \n")
    
    #создать список полных путей к каждой кардиограмме в папке
    names = os.listdir(path)
    files = [os.path.join(path, name) for name in names][:amount]
    global __cardiograms
    __cardiograms = np.zeros(0)
   
    def preparing(file:"list", name:"string"):
        print("\npreparing for {} in progress...".format(name))
        #считать с файла в numpy массив
        array = np.fromfile(file, dtype='i4', count=-1, sep='')
        #подменить "Nan" ноль
        array[np.isnan(array)] = 0   
        #нормализовать значения в пределах [-1,1]
        y = normalized(array)
        #удалить значения шума
        y = y[y!=   0.0]
        y = y[y!=-0.078]
        #threading.RLock().acquire()
        global __cardiograms
        __cardiograms = np.concatenate((__cardiograms, y), axis=0)
        #threading.RLock().release()


    for file, name in zip(files,names):
        threading.Thread(target=lambda: preparing(file, name), name= name).start()
 
    while threading.active_count() > 1:
        time.sleep(1)
    
    return __cardiograms


#поделить все сокращения по типам используя коэффициент корреляции Пирсона
def separate_by_types(systoles:"lists of systoles" = [], lengths:"list" = []):
    print("function 'separate_by_types' in progress \n")
    #создаем словарь, в который будем добавлять ключ - тип, на каждый ключ стисок подобных сокрщений
    types = {} 
    
    #проходимся по общему списку запоминая позицию
    for name, first in enumerate(systoles):
        #проходимся по общему списку и параллельно по длинам сокращенийб начиная с 2го значения
        for second, length in zip(systoles[1:], lengths[1:]):
            
            #находим коэффициент подобия между первым и вторым, третим.., сокрщением
            sameness = pearson_correlation(first, second)
            #если коэффициент подобия больше чем 0.85
            if sameness > 0.88: # <------------------------------------------------------------------------- 0.88
                #создаем имена ключей для сокращений и длин
                systoles_type   = "type_{}".format(name)
                systoles_length = "length_{}".format(name)
                #если ключ типа не существует, создаем новый тип в словаре, создаем длины типов
                if systoles_type not in types:
                    types[systoles_type]  = []
                    types[systoles_length]= []
                #добавляем сокращение в словарь, ключ(name) с таким же типом 
                types[systoles_type]  += [second] 
                #добавляем длину сокращения
                types[systoles_length]+= [length]
                #удалить из списка сокращений которое добавили в словарь по типу
                systoles.remove(second)

    return types


#возвращает имя(ключ) из словаря по которому список с самым большим количеством элементов, подразумевается что больше всего нормальных сокращений 
def sorted_systoles(arrays:"dict" = {}) -> "принимает словарь и возвращает список в котором первое значение - имя(ключ), второе - имя(ключ длин)":
    print("function 'sorted_systoles' in progress \n")
    assert(type(arrays)== dict), "from 'sorted_systoles' - you should use dict, not {}".format(type(arrays))

    systoles = []
    lengths  = []
    names    = []

    for name in arrays:
        if name.split("_")[0] == "type":
            systoles.append(arrays[name])
            names.append(name)

        elif name.split("_")[0] == "length":
            lengths.append( arrays[name])
        
        else:
            print("NAME: {} , it is look like bug!".format(name))
        
   
    return systoles, lengths, names


if __name__ == '__main__':
    y = merge_cardiograms(amount = 2)

    #найти пики с позициями по частоте array[0] - позиции, array[1] - значения R пиков 
    systoles_peak = get_peaks(cardiogram = y)

    #разбить массив на куски по сокращениям используя данные от "get_peaks" - array[0], сделать их одинаковыми размерами  
    cardiogram = separate_by_peak(offcuts = systoles_peak[0], cardiogram = y, length = 200)

    #возвращает словарь, на каждый тип свой ключ, по каждому ключу список сокращений одного типа 
    systoles_by_types = separate_by_types(systoles = cardiogram["systoles"], lengths = cardiogram["lengths"])  




    for systole, length, name in zip(*sorted_systoles(systoles_by_types)):
        size = len(systole)
        if size < 10:
            continue

        print("systole----------", name, len(systole), len(length))
        seed = random.random()
        random.shuffle(systole, lambda:seed)
        random.shuffle(length,  lambda:seed)


        buffer01 = systoles_generator(cardiogram = systole, lengths = length, size = 1000)

        save_json(buffer01["original"], r"cardiogram_{}_size_{}.json".format(name, size))
        #json_cardiogram = load_json(r"cardiogram_{}_size_{}.json".format(name, size))
        #render(sum(json_cardiogram, []))


    print("DONE!")
    ##скачивает с сервера определенное количество кардиограмм, определенных размеров
    #downloader(         url = r"https://sandbox.heartin.net/api/v1/download/cardiogram-uploads/", 
    #           list_uploads = r"list_uploads",
    #              save_path = r"binary",
    #              from_size = 1000000)
                    
    #создать GIF анимацию из массива
    #animation(array = sum(buffer, []), max_time = 100, path = r"C:\Users\o.zaitsev\Source\Repos\neuralNetwork\Heart-In\GIF\\")
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