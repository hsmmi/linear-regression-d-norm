from matplotlib import pyplot as plt
import numpy as np
from random import randint
from random import seed
import os

from my_io import build_testcase, read_dataset, read_dataset_with_pandas_to_nparray
from norm import find_error_x_from_y, find_rep_of_vector

number_of_value = 100
d_norm = 2
lenght_of_value = 100

# build_testcase(number_of_value,d_norm,lenght_of_value)
main_value = read_dataset_with_pandas_to_nparray('dataset/Data-Train-mini.csv',0)
target_value = read_dataset_with_pandas_to_nparray('dataset/Data-Train-mini.csv',-1)

number_of_value = len(main_value)
end_lenght_of_value, start_lenght_of_value = main_value.max(), main_value.min()
llenght_of_value = end_lenght_of_value - start_lenght_of_value

biasing = float(0)
weighting = float(0)

def find_biasing(main_value, target_value, d_norm):
    """
    Gets main value, target vaule and d_norm
    Biasing should be the representor of vector deviation between main value and 
    target value to minimize error
    Return representor for (target value - main value) in norm d
    """
    deviation = np.subtract(target_value, main_value)
    return find_rep_of_vector(deviation, d_norm)

def find_weighting(main_value, target_value, d_norm):
    number_of_value = len(main_value)

    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.divide(target_value, main_value)

    if (d_norm == 0):
        from statistics import mode
        return mode(weights)

    sorted_weights = [tuple((i, weights[i]))for i in range(number_of_value)]
    sorted_weights.sort(key=lambda x: x[1])

    if (d_norm == 1):
        #Using Sweap Line
        sum_of_values = sum(main_value)
        initial_error = find_error_x_from_y(main_value, target_value)
        sum_befor = 0
        sum_after = sum_of_values
        min_error = (lenght_of_value**d_norm)*number_of_value
        weighting_of_min_error = -1

        for i in range(number_of_value):
            sum_after -= main_value[sorted_weights[i][0]]
            if(abs(initial_error - (sorted_weights[i][1]-1)*(sum_befor-sum_after)) < min_error):
                min_error = abs(initial_error - (sorted_weights[i][1]-1)*(sum_befor-sum_after))
                weighting_of_min_error = sorted_weights[i][1]
            sum_after += main_value[sorted_weights[i][0]]
        return weighting_of_min_error

    if (d_norm == 2):
        #Lets use Ternary Search
        start_point = sorted_weights[0][1]
        end_point = sorted_weights[number_of_value-1][1]
        while(end_point-start_point > 1e-7):
            main_value_a = start_point+(end_point-start_point)*1/3
            main_value_b = start_point+(end_point-start_point)*2/3
            f_main_value_a = find_error_x_from_y(main_value*main_value_a, target_value)
            f_main_value_b = find_error_x_from_y(main_value*main_value_b, target_value)
            if(f_main_value_a > f_main_value_b):
                start_point = main_value_a
            elif (f_main_value_a == f_main_value_b):
                start_point = main_value_a
                end_point = main_value_b
            else:
                end_point = main_value_b
        return float(start_point)

    if (d_norm > 2):
        return ((sorted_weights[0][1]*main_value[sorted_weights[0][0]]
            + sorted_weights[number_of_value-1][1]*main_value[sorted_weights[number_of_value-1][0]])
            / (main_value[sorted_weights[0][0]]+main_value[sorted_weights[number_of_value-1][0]]))

def alternate_search(main_value, target_value, method):

    if(method == 'nod_norm'):
        itr = 0
        biasing = 0
        weighting = 1
        error_biasing = 0
        next_error = find_error_x_from_y(main_value, target_value)
        error_biasing = next_error

        if(itr % 2 == 0):
            biasing = find_biasing(main_value*weighting, target_value)
            next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)

        else:
            weighting = find_weighting(main_value+biasing, target_value, d_norm)
            next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)

        nod_normif = 0
        while (nod_normif < 100):
            if(abs(error_biasing - next_error) < 0.00001):
                nod_normif += 1
            else:
                nod_normif = 0

            error_biasing = next_error

            if(itr % 2 == 0):
                biasing = find_biasing(main_value*weighting, target_value)
                next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)

            else:
                weighting = find_weighting(main_value+biasing, target_value, d_norm)
                next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)
            
            itr += 1
        return weighting, biasing

    if(method == 'noImp'):
        itr = 0
        biasing = 0
        weighting = 1
        errBest = (lenght_of_value**d_norm)*number_of_value
        best_biasing = 0
        best_weighting = 0
        next_error = find_error_x_from_y(main_value, target_value)

        if(itr % 2 == 0):
            biasing = find_biasing(main_value*weighting, target_value)
            next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)

        else:
            weighting = find_weighting(main_value+biasing, target_value, d_norm)
            next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)

        noImp = 0
        while (noImp < 100):
            if(errBest - next_error < 0.00000001):
                noImp += 1
            else:
                noImp = 0

            if(next_error < errBest):
                errBest = next_error
                best_weighting = weighting
                best_biasing = biasing

            if(itr % 2 == 0):
                biasing = find_biasing(main_value*weighting, target_value)
                next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)

            else:
                weighting = find_weighting(main_value+biasing, target_value, d_norm)
                next_error = find_error_x_from_y(main_value*weighting+biasing, target_value)
            
            itr += 1
        return best_weighting, best_biasing


print(f'basic error is:\n{find_error_x_from_y(main_value,target_value)}\n')
weighting, biasing = alternate_search(main_value,target_value,"noImp")
print(f'final a is {weighting} and c is {biasing} with error {find_error_x_from_y(main_value*weighting+biasing,target_value)}')
# weighting, biasing = alternate_search(main_value,target_value,"nod_norm")
# print(f'final a is {weighting} and c is {biasing} with error {find_error_x_from_y(main_value*weighting+biasing,target_value)}')




# rg = np.arange(0,20,0.02)
# x = np.zeros_like(rg)
# y = np.zeros((len(rg),len(rg)))
# for ii,i in enumerate(rg):
#     for jj,j in enumerate(rg):
#         x[ii] = i
#         y[ii][jj]=find_error_x_from_y(main_value*i+j,target_value)
# xmin = int(np.argmin(y)/len(rg))
# ymin = int(np.argmin(y)%len(rg))
# print(f'min error in weights & c is {np.min(y)} in weights = {x[int(np.argmin(y)/len(rg))]} and c = {x[int(np.argmin(y)%len(rg))]}\n')
# print(find_error_x_from_y(main_value*x[xmin]+x[ymin],target_value))
# print(min(rg),max(rg))

# ax1.plot(main_value, y, '.')
# ax1.set_title('With biasing')
# ax1.set(main_valuelabel='Δ', ylabel='err')
# y2 = np.zeros(number_of_value)
# ax1.plot(deviation, y2, 'x')
# plt.show()