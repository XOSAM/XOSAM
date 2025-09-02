import numpy as np
import matplotlib.pyplot as plt
import os

def compute_gradient(function, x_range=(-50, 50.1, 0.1)):
    x = np.arange(*x_range)
    y = function(x)
    array_xy = np.column_stack((x, y))
    gradient = np.diff(y) / np.diff(x)
    return array_xy, gradient

def find_min_y(array_xy, gradient):
    y_min = array_xy[:,1].min()
    idx_min = array_xy[:,1].argmin()
    slope_before = gradient[idx_min-1] if idx_min > 0 else None
    slope_after = gradient[idx_min] if idx_min < len(gradient) else None
    return y_min, slope_before, slope_after

def f_linear(x):
    return 12*x + 1

def f_quad1(x):
    return x**2

def f_quad2(x):
    return 2*x**2 + 2*x

def f_sin(x):
    return np.sin(x/12)

functions = [f_linear, f_quad1, f_quad2, f_sin]
names = ['y = 12x + 1', 'y = x^2', 'y = 2x^2 + 2x', 'y = sin(x/12)']
ranges = [(-50, 50.1, 0.1), (-50, 50.1, 0.1), (-50, 50.1, 0]()
