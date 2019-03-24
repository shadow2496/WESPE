import os

import numpy as np
from PIL import Image


def load_train_data(dataset_dir, phone, train_size, image_size):
    """
    dataset_dir : refer to dataset path
    phone : {iphone, blackberry, sony}, 3 types of model exist
    train_size : deciding how many number of images you choose
    image_size : deciding size of image
    """
    train_dir_phone = dataset_dir + phone + '/training_data/' + phone + '/'
    train_dir_dslr = dataset_dir + phone + '/training_data/canon/'
    num_train_images = len(os.listdir(train_dir_phone))

    # Load all training images if train_size == -1
    if train_size == -1:
        train_size = num_train_images
        image_names = np.arange(0, train_size)
    else:
        image_names = np.random.choice(np.arange(0, num_train_images), train_size, replace=False)

    train_phone = np.zeros((train_size, image_size))
    train_dslr = np.zeros((train_size, image_size))

    for i, name in enumerate(image_names):
        img = np.asarray(Image.open(train_dir_phone + str(name) + '.jpg'))
        img = np.float32(img.transpose((2, 0, 1)).reshape(image_size)) / 255
        train_phone[i] = img

        img = np.asarray(Image.open(train_dir_dslr + str(name) + '.jpg'))
        img = np.float32(img.transpose((2, 0, 1)).reshape(image_size)) / 255
        train_dslr[i] = img

        if not (i + 1) % 100:
            print("Loading training data {}/{}...".format(i + 1, train_size), end="\r")

    return train_phone, train_dslr


def load_test_data(model, path, test_start, test_end, image_size):
    """
    model : {iphone, blackberry, sony}, 3 types of model exist
    path  : refer to dataset path
    image_size : deciding size of image
    """
    test_path_phone = path + str(model) + '/test_data/patches/' + str(model) + '/'
    test_path_canon = path + str(model) + '/test_data/patches/canon/'

    test_image_num = len([name for name in os.listdir(test_path_phone)
                         if os.path.isfile(os.path.join(test_path_phone, name))])

    test_size = test_end - test_start
    if test_size == -1:  # use all test image
        test_size = test_image_num
        test_image = np.arange(0, test_size)
    else:                # use small test image
        test_image = np.arange(test_start, test_end)

    test_phone = np.zeros((test_size, image_size))
    test_canon = np.zeros((test_size, image_size))

    idx = 0
    for img in test_image:
        img_array = np.asarray(Image.open(test_path_phone + str(img) + '.jpg'))
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        test_phone[idx, :] = img_array

        img_array = np.asarray(Image.open(test_path_canon + str(img) + '.jpg'))
        img_array = np.float32(np.reshape(img_array, newshape=[1, image_size])) / 255
        test_canon[idx, :] = img_array

        idx += 1
        if idx % 100 == 0:
            print('image / test_size : %d / %d = %.2f percent done' % (idx, test_size, idx/test_size))

    return test_phone, test_canon
