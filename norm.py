from math import sqrt
from statistics import mode

import numpy as np

def update_norm(preNorm, val, norm = 2):
    if(norm == 0):
        return preNorm + int(val != 0)
    elif(norm == 1):
        return preNorm + abs(val)
    elif(norm == 2):
        return sqrt(preNorm**2+val**2)
    else:
        return max(preNorm, abs(val))


def update_norm_from_rep(preNorm, val, rep, norm = 2):
    return update_norm(preNorm,val-rep,norm)

def any_to_nparray(any):
    if(type(any).__module__ != np.__name__):
        any = np.array(any)
    return any

def deviation_vector_form_rep(vec , rep, norm = 2) -> float:
    vec = any_to_nparray(vec)

    if(norm == 0):
        return sum(vec != rep)

    elif(norm == 1):
        return sum(abs(vec-rep))

    elif(norm == 2):
        return sqrt(sum((vec-rep)**2))

    else:
        return max(abs(rep - vec.min()),abs(rep - vec.max()))

def find_rep_of_vector(vec , norm = 2):
    vec = any_to_nparray(vec)
    if(norm == 0):
        return mode(vec)

    elif(norm == 1):
        return np.median(vec)

    elif(norm == 2):
        return np.mean(vec)

    else:
        return (vec.max() + vec.min()) / 2

def norm_of_vector(vec , norm = 2) -> float:
    vec = any_to_nparray(vec)

    if(vec.size == 1):
        return update_norm(0,vec,norm)

    if(norm == 0):
        return sum(vec != 0)

    elif(norm == 1):
        return sum(abs(vec))

    elif(norm == 2):
        return sqrt(sum(vec**2))

    else:
        return max(abs(vec.min()),abs(vec.max()))


def find_error_x_from_y(main_value, target_value, d_norm):
    return norm_of_vector(main_value-target_value, d_norm)
