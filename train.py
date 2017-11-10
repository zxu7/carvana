import os, glob, cv2, argparse
import numpy as np
import tensorflow as tf
from multiprocessing.dummy import Pool
from datetime import datetime
from utils import *
from models import *
from keras.optimizers import Adam, rmsprop
from keras.callbacks import ModelCheckpoint, CSVLogger
from math import ceil
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.utils.training_utils import multi_gpu_model

img_dir = os.path.join(os.path.expanduser('~'), 'school/data science/kaggle/carvana/train')
mask_dir = os.path.join(os.path.expanduser('~'), 'school/data science/kaggle/carvana/train_masks')


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_imdir', type=str, default=img_dir)
    parser.add_argument('--train_maskdir', type=str, default=mask_dir)
    parser.add_argument('--target_size', type=tuple, default=(256,256))
    parser.add_argument('--grayscale', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', type=float, default=1)
    parser.add_argument('--gpus', type=str, default=None, help="gpu1 use '1'; multi-gpu training use '0,1';")
    return parser.parse_args()


# train
if __name__ == "__main__":
    args = create_args()
    # construct train, valid datasets
    img_dir = args.train_imdir
    mask_dir = args.train_maskdir
    target_size = args.target_size
    grayscale = args.grayscale
    batch_size = args.batch_size
    epochs = args.epochs


    now = datetime.now()
    if grayscale:
        target_size = target_size + (1,)
    else:
        target_size = target_size + (3,)
    filepath_dir = 'experiment/model-{}-{}-{}-{}-{}-{}-{}/'.format(now.month, now.day, now.hour, now.minute, target_size[0],
                                                                target_size[1], target_size[2])
    if not os.path.isdir(filepath_dir):
        os.makedirs(filepath_dir)
    # filepath = filepath_dir + 'epoch-{epoch:02d}-{loss:.4f}-{dice_coef:.4f}-{val_loss:.4f}-{val_dice_coef:.4f}.hdf5'
    filepath = filepath_dir + 'best_model.hdf5'

    # construct fn dictionary
    total_fns = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(os.path.join(img_dir, '*.jpg'))]
    fn_dict = {fn: [os.path.join(img_dir, fn + '.jpg'), os.path.join(mask_dir, fn + '_mask.gif')] for fn in total_fns}
    # train/valid split
    np.random.seed(1)
    valid_fns = list(np.random.choice(total_fns, 300))
    train_fns = [x for x in total_fns if x not in valid_fns]

    # normalize = normalize_data(train_fns, fn_dict, target_size)
    normalize = None

    # load model and train
    # config gpu% use
    if args.gpu:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu
        if len(args.gpus.split(',')) == 1:
            config.gpu_options.visible_device_list = args.gpus
        set_session(tf.Session(config=config))

    train_gen = data_gen(train_fns, fn_dict, shuffle=True)
    valid_gen = data_gen(valid_fns, fn_dict)

    # single-gpu training
    if len(args.gpus.split(',')) == 1:
        # model = SimpleCNN(target_size, normalize=normalize)
        model = unet(target_size, normalize=normalize)
        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, monitor='val_dice_coef',
                                       mode='max')
        csvlogger = CSVLogger(filepath_dir + 'training.log')
        model.compile(loss=bce_dc_loss, optimizer=rmsprop(1e-4), metrics=[dice_coef, 'accuracy'])
        try:
            model.fit_generator(train_gen, steps_per_epoch=ceil(len(train_fns)/batch_size), epochs=epochs,
                                validation_data=valid_gen, validation_steps=ceil(len(valid_fns)/batch_size),
                                callbacks=[checkpointer, csvlogger])
        except AttributeError:
            pass

    # multi-gpu training
    elif len(args.gpus.split(',')) > 1:
        with tf.device("/cpu:0"):
            # model = SimpleCNN(target_size, normalize=normalize)
            model = unet(target_size, normalize=normalize)
            multi_model = multi_gpu_model(model, len(args.gpus.split(',')))
        # TODO: ModelCheckpoint does not work for multi_gpu_model yet. Make one that works!
        # checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True, monitor='val_dice_coef',
        #                                mode='max')
        csvlogger = CSVLogger(filepath_dir + 'training.log')
        model.compile(loss=bce_dc_loss, optimizer=rmsprop(1e-4), metrics=[dice_coef, 'accuracy'])
        multi_model.compile(loss=bce_dc_loss, optimizer=rmsprop(1e-4), metrics=[dice_coef, 'accuracy'])
        try:
            multi_model.fit_generator(train_gen, steps_per_epoch=ceil(len(train_fns)/batch_size), epochs=epochs,
                                validation_data=valid_gen, validation_steps=ceil(len(valid_fns)/batch_size),
                                callbacks=[csvlogger])
        except AttributeError:
            pass
        model.set_weights(multi_model.get_weights())
        model.save(filepath=filepath)

    else:
        raise NotImplementedError("--gpus argument not understood. argument has to have form '0,1' or '1' ")


