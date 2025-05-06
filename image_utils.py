import os
import random
import tempfile
import urllib.error
import urllib.parse
import urllib.request

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torchvision  # use for loading CIFAR10


def make_grid(array, ncols=3, padding=0):
    nindex, height, width, intensity = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    grid = array.reshape(nrows, ncols, height, width, intensity)
    if padding > 0:
        assert padding % 2 == 0
        padded = np.full((nrows, ncols, height+padding, width+padding, intensity), array.max())
        p = padding // 2
        padded[:, :, p:-p, p:-p] = grid
        grid = np.ascontiguousarray(padded)
        height += padding
        width += padding
    result = (grid
              .swapaxes(1,2)
              .reshape(height * nrows, width * ncols, intensity))
    return result


def process_mnist(path, visualize=True, fully_connected=False):
    mnist = np.load(os.path.join(path, 'mnist.npz'))
    x_train = mnist['x_train']
    y_train = mnist['y_train']
    x_test = mnist['x_test']
    y_test = mnist['y_test']

    num_test = len(y_test)

    if visualize:
        samples_per_class = 10
        samples = []
        np.random.seed(0)
        random.seed(0)

        for y in range(10):
            idxs = np.nonzero(y_train == y)[0]
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])]
                samples.append(x_train[idx])
        img = make_grid(np.expand_dims(np.array(samples), -1), ncols=samples_per_class)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    if fully_connected:
        # data preprocessing for neural network with fully-connected layers
        data = {
            'X_train': np.array(x_train[:55000], np.float32).reshape((55000, -1)),  # training data
            'y_train': np.array(y_train[:55000], np.int32),  # training labels
            'X_val': np.array(x_train[55000:], np.float32).reshape((5000, -1)),  # validation data
            'y_val': np.array(y_train[55000:], np.int32),  # validation labels
            'X_test': x_test.reshape((num_test, -1)),  # test data
            'y_test': y_test,  # test labels
        }
    else:
        # data preprocessing for neural network with convolutional layers
        data = {
           'X_train': np.array(x_train[:55000], np.float32).reshape((55000, 1, 28, 28)),  # training data
           'y_train': np.array(y_train[:55000], np.int32),  # training labels
           'X_val': np.array(x_train[55000:], np.float32).reshape((5000, 1, 28, 28)),  # validation data
           'y_val': np.array(y_train[55000:], np.int32),  # validation labels
           'X_test': np.expand_dims(x_test, 1), # test data
           'y_test': y_test,  # test labels
        }
    return data


def process_cifar(path, visualize=True):
    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True)

    D_train = (trainset.data.transpose(0, 3, 1, 2).astype(np.float32) / 256.) - 0.5
    D_test = (testset.data.transpose(0, 3, 1, 2).astype(np.float32) / 256.) - 0.5
    data = {
        'classes': trainset.classes,
        'X_train': D_train[:45000],
        'y_train': np.array(trainset.targets[:45000]).astype(np.int32),
        'X_val': D_train[45000:],
        'y_val': np.array(trainset.targets[45000:]).astype(np.int32),
        'X_test': D_test,
        'y_test': np.array(testset.targets).astype(np.int32),
    }

    if visualize:
        samples_per_class = 10
        samples = []
        np.random.seed(0)
        random.seed(0)

        for y in range(10):
            idxs = np.nonzero(data['y_train'] == y)[0]
            for i in range(samples_per_class):
                idx = idxs[random.randrange(idxs.shape[0])]
                samples.append(data['X_train'][idx])
        img = make_grid(np.array(samples).transpose(0, 2, 3, 1) + 0.5, ncols=samples_per_class)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    return data


def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imageio.imread(fname)
        #If there is an error from "os.remove(fname)". Please try commenting
        #out this line and running the code. The code then won't delete the
        #downloaded file for you, but you can delete it manually later.
        #os.remove(fname)
        print('Successfully get image from', url)
        return img
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)
        raise
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
        raise
