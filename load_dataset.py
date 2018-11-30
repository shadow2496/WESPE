from __future__ import division
from PIL import Image

import numpy as np
import os

# load test and train dataset
# path default = './dataset/dped/'

def load_train_dataset(model, path, train_size, image_size):
    """
    model : {iphone, blackberry, sony}, 3 types of model exist
    path  : refer to dataset path
    train_size : deciding how many number of images you choose
    image_size : deciding size of image
    """
    train_path_phone = path + str(model) + '/training_data/' + str(model) + '/'
    train_path_canon  = path + str(model) + '/training_data/canon/'

    train_image_num = len([name for name in os.listdir(train_path_phone) \
                           if os.path.isfile(os.path.join(train_path_phone, name))])

    if train_size == -1: # use all training image
        train_size  = train_image_num
        train_image = np.arange(0, train_size)
    else:                # use small training image
        train_image = np.random.choice(np.arange(0, train_image_num), size=train_size, replace=False)

    train_phone = np.zeros((train_size, image_size))
    train_canon = np.zeros((train_size, image_size))

    idx = 0
    for img in train_image:
        img_array = np.asarray(Image.open(train_path_phone + str(img) + '.jpg'))    # (100, 100, 3), 0~255 value
        img_array = np.float32(np.reshape(img_array, newshape=[1,image_size])) / 255
        train_phone[idx, :] = img_array

        img_array = np.asarray(Image.open(train_path_canon + str(img) + '.jpg'))    # (100, 100, 3), 0~255 value
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        train_canon[idx, :] = img_array

        idx += 1
        if idx % 100 == 0:
            print ('image / train_size : %d / %d = %.2f percent done' % (idx, train_size, idx/train_size))

    return train_phone, train_canon


def load_test_dataset(model, path, image_size):
    """
    model : {iphone, blackberry, sony}, 3 types of model exist
    path  : refer to dataset path
    image_size : deciding size of image
    """
    test_path_phone = path + str(model) + './test_data/patches/' + str(model) + '/'
    test_path_canon = path + str(model) + './test_data/patches/canon/'

    test_image_num = len([name for name in os.listdir(test_path_phone) \
                         if os.path.isfile(os.path.join(test_path_phone, name))])

    test_phone = np.zeros((test_image_num, image_size))
    test_canon = np.zeros((test_image_num, image_size))

    for img in range(test_image_num):
        img_array = np.asarray(Image.open(test_path_phone + str(img) + '.jpg'))
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        test_phone[img, :] = img_array

        img_array = np.asarray(Image.open(test_path_canon + str(img) + '.jpg'))
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        test_canon[img, :] = img_array

        if img % 100 == 0:
            print ('image / test_size : %d / %d = %.2f percent done' % (img, test_image_num, img/test_image_num))

    return test_phone, test_canon
