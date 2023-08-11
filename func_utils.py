import numpy as np
import matplotlib.pyplot as plt
import sys
import flap
import time

def derivative_function(input_array):
    result = [0]
    for i in range(len(input_array)-1):
        result.append(input_array[i+1] - input_array[i])
    return np.array(result)

def arr_normalize(input_array):
    max_val = np.amax(input_array)
    if type(input_array) is list:
        result = np.array(input_array)
    else: 
        result = input_array
    normalized_result = result/max_val
    return normalized_result

def merge_default_and_input_dict(input_dict, default_dict):
    if input_dict == None:
            return default_dict
    else:
        result_dict = default_dict
        for i in result_dict:
            if i in input_dict:
                result_dict[i] = input_dict[i]
    return result_dict

#decorator functions:
def stopwatch(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        funcval = func(*args, **kwargs)
        end = time.time()
        print('Execution time: ' + str(end - start))
        return funcval
    return wrapper
