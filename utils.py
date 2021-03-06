import os
import glob
import cv2
import keras
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from PIL import Image
from keras.preprocessing.image import load_img
from keras.callbacks import Callback, warnings
from config import ORIGIN_SHAPE
from multiprocessing.dummy import Pool as ThreadPool


class MultiGPUModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

# threadsafe generator
class DataIterator(object):
    def __init__(self,fns, fn_dict, target_size=(256,256), grayscale=True, batch_size=2, data_aug=False, shuffle=False,
             test=False):
        self.fns = fns
        self.fn_dict = fn_dict
        self.target_size = target_size
        self.grayscale = grayscale
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.shuffle = shuffle
        self.test = test
        self.lock = threading.Lock()
        self.datagenerator = self.data_gen()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            x,y = next(self.datagenerator)
        return x, y

    def data_gen(self):
        idx = 0
        batch_x = np.zeros((self.batch_size,) + self.target_size + (3,))
        batch_y = np.zeros((self.batch_size,) + self.target_size + (1,))
        if self.shuffle:
            fns = np.random.permutation(self.fns)
        if self.grayscale:
            batch_x = np.zeros((self.batch_size,) + self.target_size + (1,))
        while True:
            if not self.test:
                for i, fn in enumerate(self.fns[idx: (idx + self.batch_size)]):
                    img = load_img(self.fn_dict[fn][0], grayscale=self.grayscale, target_size=self.target_size)
                    mask = load_img(self.fn_dict[fn][1], grayscale=True, target_size=self.target_size)
                    if self.grayscale:
                        batch_x[i] = np.array(img, dtype='float32')[:, :, np.newaxis]
                    else:
                        batch_x[i] = np.array(img, dtype='float32')
                    batch_y[i] = np.array(mask, dtype='float32')[:, :, np.newaxis]
                idx += self.batch_size
                yield batch_x, batch_y / 255.
            else:
                for i, fn in enumerate(self.fns[idx: (idx + self.batch_size)]):
                    img = load_img(self.fn_dict[fn][0], grayscale=self.grayscale, target_size=self.target_size)
                    if self.grayscale:
                        batch_x[i] = np.array(img, dtype='float32')[:, :, np.newaxis]
                    else:
                        batch_x[i] = np.array(img, dtype='float32')
                idx += self.batch_size
                yield batch_x, None


class DataGenerator(DataIterator):
    def __init__(self, **kwargs):
        super(DataGenerator, self).__init__(**kwargs)

    def __next__(self):
        idx = 0
        batch_x = np.zeros((self.batch_size,) + self.target_size + (3,))
        batch_y = np.zeros((self.batch_size,) + self.target_size + (1,))
        if self.shuffle:
            fns = np.random.permutation(self.fns)
        if self.grayscale:
            batch_x = np.zeros((self.batch_size,) + self.target_size + (1,))
        while True:
            if not self.test:
                for i, fn in enumerate(self.fns[idx: (idx + self.batch_size)]):
                    img = load_img(self.fn_dict[fn][0], grayscale=self.grayscale, target_size=self.target_size)
                    mask = load_img(self.fn_dict[fn][1], grayscale=True, target_size=self.target_size)
                    if self.grayscale:
                        batch_x[i] = np.array(img, dtype='float32')[:, :, np.newaxis]
                    else:
                        batch_x[i] = np.array(img, dtype='float32')
                    batch_y[i] = np.array(mask, dtype='float32')[:, :, np.newaxis]
                idx += self.batch_size
                return batch_x, batch_y / 255.
            else:
                for i, fn in enumerate(self.fns[idx: (idx + self.batch_size)]):
                    img = load_img(self.fn_dict[fn][0], grayscale=self.grayscale, target_size=self.target_size)
                    if self.grayscale:
                        batch_x[i] = np.array(img, dtype='float32')[:, :, np.newaxis]
                    else:
                        batch_x[i] = np.array(img, dtype='float32')
                idx += self.batch_size
                return batch_x


def rle_decode(x, shape=(1918, 1280)):
    width, height = shape
    n = width*height
    x = x.split()
    img = np.zeros(n, dtype='uint8')
    pos_nums = list(zip(x[::2], x[1::2]))
    for pn in pos_nums:
        pos, num = pn
        pos = int(pos) - 1 # rle index starts at 1, matrix index starts at 0
        num = int(num)
        img[pos:(pos+num)] = 1
    return img.reshape((height, width))


def rle_encode(x, mode='faster'):
    x = np.array(x.flatten() > 0.5, dtype=int)
    # ones = np.where(x==1)[0] + 1 # index starts at 1
    out = []
    if mode == 'faster':
        x[0] = 0
        x[-1] = 0
        runs = np.where(x[1:] != x[:-1])[0] + 2
        runs[1::2] = runs[1::2] - runs[:-1:2]

    elif mode == 'fast':
        ones = np.where(x == 1)[0]
        prev = -2
        for b in ones:
            if b > prev + 1: out.extend((b + 1, 0))
            out[-1] += 1
            prev = b
        out = [str(x) for x in out]
    else:
        out = []
        cnt = 0
        for i, x1 in enumerate(x):
            if x1 == 1 and x[i - 1] == 0:
                cnt += 1
                out.append(str(i + 1))
            elif x1 == 1 and x[i - 1] == 1:
                cnt += 1
            elif x1 == 0 and x[i - 1] == 1:
                out.append(str(cnt))
                cnt = 0
            else:
                pass
    return ' '.join(out)


def dice_coef(y_true, y_pred):
    y_pred = K.cast(K.greater_equal(y_pred, 0.5), dtype='float32')
    num = K.sum(2*y_true * y_pred, axis=[1,2,3]) + 1e-5
    denom = K.sum(K.cast(K.equal(y_true, 1), dtype='float32') + K.cast(K.equal(y_pred, 1),dtype='float32'), axis=[1,2,3]) + 1e-5
    out = num/denom
    return K.mean(out, axis=-1)


def bce_dc_loss(y_true, y_pred):
    return keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


def normalize_data(fns, fn_dict, target_size):
    '''
    :param fns: filenames
    :param target_size: tuple; (x, y, channel)
    :return: channelwise normalization
    '''
    n = len(fns)
    x, y, channel = target_size
    grayscale = True
    if channel > 1:
        grayscale = False
    pool = ThreadPool(5)

    def map_func(fn):
        img_dir = fn_dict[fn][0]
        out = np.array(load_img(img_dir, target_size=(x, y), grayscale=grayscale), dtype='float32')[np.newaxis,:]
        if grayscale:
            out = out[:,:,:,np.newaxis]
        return out

    data = pool.map(map_func, fns)
    pool.close()
    pool.join()
    data = np.concatenate(data)
    print("data normalized for {} images...".format(n))
    return np.mean(data, (0,1,2))


def data_gen(fns, fn_dict, target_size=(256,256), grayscale=True, batch_size=2, data_aug=False, shuffle=False,
             test=False):
    '''
    :param fns: list of filenames
    :param fn_dict: {'fn':['path/to/img', 'path/to/mask']}
    :return:
    '''
    lock = threading.Lock()
    with lock:
        idx = 0
        batch_x = np.zeros((batch_size, ) + target_size + (3, ))
        batch_y = np.zeros((batch_size, ) + target_size + (1, ))
        if shuffle:
            fns = np.random.permutation(fns)
        if grayscale:
            batch_x = np.zeros((batch_size, ) + target_size + (1,))
        while True:
            if not test:
                for i, fn in enumerate(fns[idx : (idx + batch_size)]):
                    img = load_img(fn_dict[fn][0], grayscale=grayscale, target_size=target_size)
                    mask = load_img(fn_dict[fn][1], grayscale=True, target_size=target_size)
                    if grayscale:
                        batch_x[i] = np.array(img, dtype='float32')[:,:,np.newaxis]
                    else:
                        batch_x[i] = np.array(img, dtype='float32')
                    batch_y[i] = np.array(mask, dtype='float32')[:,:,np.newaxis]
                idx += batch_size
                yield batch_x, batch_y/255.
            else:
                for i, fn in enumerate(fns[idx : (idx + batch_size)]):
                    img = load_img(fn_dict[fn][0], grayscale=grayscale, target_size=target_size)
                    if grayscale:
                        batch_x[i] = np.array(img, dtype='float32')[:,:,np.newaxis]
                    else:
                        batch_x[i] = np.array(img, dtype='float32')
                idx += batch_size
                yield batch_x


def parse_model_name(x):
    xs = x.split('-')
    target_size = (int(xs[-3]), int(xs[-2]))
    grayscale = True
    channel = int(xs[-1])
    if channel > 1: grayscale = False
    return target_size, grayscale


def resize_mask_matrix(x, size):
    '''
    :param x: mask matrix: [x, y, c]
    :return: resized mask matrix
    '''
    x = np.uint8(x[:,:,-1] > 0.5)*255
    x_im = Image.fromarray(x)
    out = np.array(x_im.resize(size)) / 255.
    return out[:, :, np.newaxis]


def resize_mask_matrix_encode(x, size=ORIGIN_SHAPE):
    x_resize = resize_mask_matrix(x, size)
    return rle_encode(x_resize, mode='faster')
