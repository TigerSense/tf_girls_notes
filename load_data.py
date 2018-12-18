import tensorflow as tf
from scipy.io import loadmat as load
from matplotlib import pyplot as plt

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
    label = np.array
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

def label_distribution(labels):
    pass
    
    
    
    
def show_image(dataset, labels, i):
    plt.imshow(dataset[i])
    plt.show()
    




             


