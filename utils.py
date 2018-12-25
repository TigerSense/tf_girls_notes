import numpy as np

def rgb2gray(data):
    coef = np.array([0.2989, 0.5870, 0.1140])
    data = np.tensordot(data, coef, axes=([3],[0]))
    return data

def normalize(data, min, max):
    (data/255)*(max-min) + min
    return data.astype(dtype=np.float32)
    
    
def one_hot_coding(labels):
    labels = labels % 10
    return np.eye(10)[labels]
    
