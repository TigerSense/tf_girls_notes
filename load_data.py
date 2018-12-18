import tensorflow as tf
from scipy.io import loadmat as load
from matplotlib import pyplot as plt
import numpy as np

train_data = load('train_32x32.mat')
test_data = load('test_32x32.mat')
extra_data = load('extra_32x32.mat')


print('train_data shape:', train_data['X'].shape)
print('train_data label shape', train_data['y'].shape)

print('test_data shape:', test_data['X'].shape)
print('test_data label shape', test_data['y'].shape)

train_samples = train_data['X']
train_labels = train_data['y']
test_samples = test_data['X']
test_labels = test_data['y']

num_labels = 10
image_size = 32

def reformat(dataset, label):
    pass
    # orignal dimension: height x width x channel x num_samples
    # reshape to: num_samples x height x width x channel
    num_samples = dataset.shape[3]
    num_classes = 10
    dataset = np.transpose(dataset,(3,0,1,2))
    
    # generate one-hot coding label
    label = np.mod(label, num_classes)
    one_hot = np.eye(num_classes)[label]
    return dataset, one_hot

def normalize(samples):
    # sample shape : num_samples x im_size x im_size x 3
    # change image from rgb to gray    
    coef = [0.299, 0.587, 0.114]
    gray = np.dot(samples, coef)

    # change pixel range [0 255] to range [-1 1]
    def normalize(a, i):
        a[i] = a[i] / 255 * 2 -1
    return a
        
    normalized = np.apply_over_axes(normalize,gray,[1,2])

def label_distribution(labels, title):
    label_sum = np.sum(labels,0)
    labels = np.arange(0,10)
    fig,ax = plt.subplots(1,1)
    ax.bar(labels,label_sum.reshape(-1))
    ax.set_title(title)
    ax.set_xlabel('label')
    ax.set_ylabel('count')
    return fig
    
    
def show_image(dataset, labels, i):
    plt.imshow(dataset[i])
    plt.show()
    

if __name__=='__main__':
    _, train_one_hot = reformat(train_samples, train_labels)
    fig = label_distribution(train_one_hot,'label distribution in train samples')
    fig.show()
    
    _, test_one_hot = reformat(test_samples, test_labels)
    fig2 = label_distribution(test_one_hot, 'label distribution in test samples')
    fig2.show()

    

 
    
    


             


