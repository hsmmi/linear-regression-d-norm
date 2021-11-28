from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
from random import randint
from random import seed
import os

from my_io import build_testcase, read_dataset, read_dataset_with_pandas_to_nparray, read_testcase
from norm import find_error_x_from_y, find_rep_of_vector

def find_biasing(main_value, target_value, d_norm):
    """
    Gets main value, target vaule and d_norm
    Biasing should be the representor of vector deviation between main value and 
    target value to minimize error
    Return representor for (target value - main value) in norm d
    """
    deviation = np.subtract(target_value, main_value)
    return find_rep_of_vector(deviation, d_norm)

def ternary_search(main_value, target_value, cost_function, start_point, end_point):
    while(end_point-start_point > 1e-6):
        one_third = start_point+(end_point-start_point)*1/3
        two_third = start_point+(end_point-start_point)*2/3
        f_one_third = cost_function(main_value*one_third, target_value, d_norm)
        f_two_third = cost_function(main_value*two_third, target_value, d_norm)
        if(f_one_third > f_two_third):
            start_point = one_third
        elif (f_one_third == f_two_third):
            start_point = one_third
            end_point = two_third
        else:
            end_point = two_third
    return float(start_point)

def sweap_line(main_value, target_value, cost_function, sorted_weights):
    number_of_value = len(main_value)
    sum_of_values = sum(main_value)
    weighting_1_error = cost_function(main_value, target_value, d_norm) # weighting = 1
    sum_befor = 0
    sum_after = sum_of_values
    min_error = weighting_1_error
    weighting_of_min_error = 1

    for i in range(number_of_value):
        current_value = main_value[sorted_weights[i][0]]
        current_weighting = sorted_weights[i][1]
        sum_after -= current_value
        current_error = abs(weighting_1_error - (current_weighting-1)*(sum_befor-sum_after))
        if(current_error < min_error):
            min_error = current_error
            weighting_of_min_error = current_weighting
        sum_after += current_value
    return weighting_of_min_error

def find_weighting(main_value, target_value, d_norm):
    number_of_value = len(main_value)

    with np.errstate(divide='ignore', invalid='ignore'):
        main_value[main_value == 0] = 1e-4
        weights = np.divide(target_value, main_value)

    if (d_norm == 0):
        from statistics import mode
        return mode(weights)

    sorted_weights = [tuple((i, weights[i]))for i in range(number_of_value)]
    sorted_weights.sort(key=lambda x: x[1])

    if (d_norm == 1):
        return sweap_line(main_value, target_value, find_error_x_from_y, sorted_weights)
    
    min_weight = sorted_weights[0][1]
    max_weight = sorted_weights[number_of_value-1][1]
    
    if (d_norm == 2):
        return ternary_search(main_value, target_value, find_error_x_from_y, min_weight, max_weight)
    
    value_of_min_weight = main_value[sorted_weights[0][0]]
    value_of_max_weight = main_value[sorted_weights[number_of_value-1][0]]
    
    if (d_norm > 2):
        return ((min_weight*value_of_min_weight + max_weight*value_of_max_weight)
            / (value_of_min_weight+value_of_max_weight))

def alternate_search(main_value, target_value):
    number_of_value = len(main_value)

    itr = 0
    biasing = 0
    weighting = 1
    best_error = np.inf
    best_biasing = 0
    best_weighting = 1
    cur_error = best_error

    noImp = 0
    while (noImp < sqrt(number_of_value)):
        if(best_error - cur_error < 1e-5):
            noImp += 1
        else:
            noImp = 0

        if(cur_error < best_error):
            best_error = cur_error
            best_weighting = weighting
            best_biasing = biasing

        if(itr % 2 == 0):
            weighting *= find_weighting(main_value*weighting+biasing, target_value, d_norm)
            cur_error = find_error_x_from_y(main_value*weighting+biasing, target_value, d_norm)

        else:
            biasing += find_biasing(main_value*weighting+biasing, target_value, d_norm)
            cur_error = find_error_x_from_y(main_value*weighting+biasing, target_value, d_norm)

        itr += 1
    return best_weighting, best_biasing

# while(1):
number_of_value = 100
d_norm = 2
lenght_of_value = 1000

build_testcase(number_of_value,d_norm,lenght_of_value)
# main_value, target_value, d_norm = read_testcase('dataset/mathAIH02.txt',0)
# plt.plot(main_value,target_value,'.')
# plt.show()
main_value = read_dataset_with_pandas_to_nparray('dataset/Data-Train.csv',0)
target_value = read_dataset_with_pandas_to_nparray('dataset/Data-Train.csv',-1)

number_of_value = len(main_value)
end_lenght_of_value, start_lenght_of_value = main_value.max(), main_value.min()
llenght_of_value = end_lenght_of_value - start_lenght_of_value

biasing = float(0)
weighting = float(0)

print(f'basic error is:\n{find_error_x_from_y(main_value,target_value,d_norm)}\n')
weighting, biasing = alternate_search(main_value,target_value)
print(f'final a is {weighting} and c is {biasing} with error {find_error_x_from_y(main_value*weighting+biasing,target_value, d_norm)}')
