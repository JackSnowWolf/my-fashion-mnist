import os
import gzip
import numpy as np


def data_load(path, kind="train", partition=None):
    """
    load fashion minist, partition is required by "t10k" testing data set
    :param path: prefix of data set
    :param kind: "train" or "t10k"
    :param partition: generate train and validation data set; None represents only generating train data set.
    :return: images and corresponding labels
    """
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as img_file:
        images = np.frombuffer(img_file.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    if partition is None:
        return images, labels
    else:
        # the size of validation set should be divisible by batch size.
        batch_size = 1000
        val_len = int(np.floor(labels.size * partition / batch_size) * batch_size)
        val_index = np.random.choice(np.arange(labels.size), val_len, replace=False)
        train_index = np.array(list(set(range(labels.size)) - set(val_index.tolist())))
        train_labels = labels[train_index]
        train_images = images[train_index]
        val_labels = labels[val_index]
        val_images = images[val_index]
        return train_images, train_labels, val_images, val_labels


if __name__ == '__main__':
    data_load(path="data/fashion", kind="train", partition=0.17)
    print("done")
