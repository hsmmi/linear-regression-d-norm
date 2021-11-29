from math import sqrt
from matplotlib import pyplot as plt
import numpy as np

from norm import find_error_x_from_y, find_rep_of_vector

def find_biasing(main_value, target_value, d_norm):
    """
    Gets main value, target vaule, and d_norm
    Biasing should be the representor of vector deviation between main value and 
    target value to minimize error
    Return representor for (target value - main value) in norm d
    """
    deviation = np.subtract(target_value, main_value)
    return find_rep_of_vector(deviation, d_norm)

def ternary_search(main_value, target_value, d_norm, cost_function, start_point, end_point):
    """
    It gets main value, target value, d_norm, cost_function, start point,
    and end point and find the best point which minimize cost function in
    our interval
    Return the best point in interval[start, end] 
    """
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

def sweep_line(main_value, target_value, d_norm, cost_function, sorted_weights):
    """
    It gets main value, target value, d_norm, cost_function, and sorted weights
    and sweep from minimum to maximum of those weights and in each weight
    calculate cost function which we can find it o(1) 
    Return argmin cost function which is the best weighting in sorted weights
    """
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
    """
    Gets main value, target vaule, and d_norm then find the best weighting
    which minimize error main value from target value
    Return representor for (target value - main value) in norm d
    """
    number_of_value = len(main_value)

    with np.errstate(divide='ignore', invalid='ignore'):
        target_value = target_value[main_value != 0]
        main_value = main_value[main_value != 0]
        number_of_value = len(main_value)
        weights = np.divide(target_value, main_value)

    if (d_norm == 0):
        from statistics import mode
        return mode(weights)

    sorted_weights = [tuple((i, weights[i]))for i in range(number_of_value)]
    sorted_weights.sort(key=lambda x: x[1])

    if (d_norm == 1):
        return sweep_line(main_value, target_value, d_norm, find_error_x_from_y, sorted_weights)
    
    min_weight = sorted_weights[0][1]
    max_weight = sorted_weights[number_of_value-1][1]
    
    if (d_norm == 2):
        return ternary_search(main_value, target_value, d_norm, find_error_x_from_y, min_weight, max_weight)
    
    value_of_min_weight = main_value[sorted_weights[0][0]]
    value_of_max_weight = main_value[sorted_weights[number_of_value-1][0]]
    
    if (d_norm > 2):
        return ((min_weight*value_of_min_weight + max_weight*value_of_max_weight)
            / (value_of_min_weight+value_of_max_weight))

def alternate_search(main_value, target_value, d_norm, printer = 0, plotter = 0):
    """
    It gets main value, target value, and d_norm then find best weighting
    and biasing which minimize the error main value from target value in
    d-norm 
    Return weighting and biasing
    """
    number_of_value = len(main_value)

    epoch = 0
    biasing = 0
    weighting = 1
    best_error = np.inf
    best_biasing = 0
    best_weighting = 1
    cur_error = best_error

    noImp = 0
    while (noImp < 10 and epoch < 1e2):

        if(epoch % 2 == 0):
            weighting *= find_weighting(main_value*weighting+biasing, target_value, d_norm)
            cur_error = find_error_x_from_y(main_value*weighting+biasing, target_value, d_norm)

        else:
            biasing += find_biasing(main_value*weighting+biasing, target_value, d_norm)
            cur_error = find_error_x_from_y(main_value*weighting+biasing, target_value, d_norm)

        epoch += 1

        if(best_error - cur_error < 1e-5):
            noImp += 1
        else:
            noImp = 0

        if(cur_error < best_error):
            best_error = cur_error
            best_weighting = weighting
            best_biasing = biasing

        if(printer):
            print((f'In epoch {round(epoch,6)} best error is {round(best_error,6)}, current error is\n'
                f'{round(cur_error,6)} with weighing {round(weighting,6)} and biasing {round(biasing,6)}\n'
                f'and no improvment is {noImp}\n'))
        if(plotter):
            plt.plot(main_value,target_value,'.')
            min_value = main_value.min()
            max_value = main_value.max()
            plt.plot([min_value,max_value], [min_value*weighting+biasing, max_value*weighting+biasing],'-')
            plt.show()
    return best_weighting, best_biasing
