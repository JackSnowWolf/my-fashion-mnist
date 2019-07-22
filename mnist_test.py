import os
import time
import numpy as np
from optparse import OptionParser
import tensorflow as tf
import glog as logger

import data_loader
import network


def compute_acc(preds, labels):
    acc = np.size(np.where(preds == labels)) / preds.size
    return acc


def my_parser():
    """
    parse arguments
    :return: options
    """
    parser = OptionParser()
    parser.add_option("-t", "--type", action="store",
                      dest="type",
                      default="CNN",
                      type="string",
                      help="set cnn or dnn")
    parser.add_option("--batch_size", action="store", dest="batch_size",
                      type="int", default=1,
                      help="set batch size")
    parser.add_option("-w", "--weight", action="store", dest="weight_path",
                      type="string",
                      default=None,
                      help="path to model weight")

    options, _ = parser.parse_args()
    return options


def cnn_test(batch_size, weight_path):
    # load data
    test_data, test_labels = data_loader.data_load("data/fashion", kind="t10k")
    assert len(test_data) == len(test_labels)

    # set configuration

    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name="input")
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="labels")
    cnn_net = network.CNNNet(phase="test", num_classes=10)

    _, pred = cnn_net.loss(labels=labels, input_data=input_layer)

    saver = tf.train.Saver()

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=sess_config)

    # start training
    with sess.as_default():
        if weight_path is None:
            raise ValueError("weight path doesn't configured")

        logger.info("Restore model from {:s}".format(weight_path))
        saver.restore(sess=sess, save_path=weight_path)
        test_acc_list = []

        # omit the last few data that less than a batch size
        batch_num = int(np.floor(test_labels.size) / batch_size)

        for i in range(batch_num):
            data_batch = test_data[i * batch_size:(i + 1) * batch_size]
            labels_batch = test_labels[i * batch_size:(i + 1) * batch_size]
            pred_label = sess.run(pred, feed_dict={input_layer: data_batch,
                                                   labels: labels_batch})
            test_acc = compute_acc(preds=pred_label, labels=labels_batch)
            test_acc_list.append(test_acc)
        acc = sum(test_acc_list) / len(test_acc_list) * 100
        logger.info("test acc: {:.2f}%%".format(acc))


def dnn_test(batch_size, weight_path):
    # load data
    test_data, test_labels = data_loader.data_load("data/fashion", kind="t10k")
    assert len(test_data) == len(test_labels)

    # set configuration

    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name="input")
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="labels")
    dnn_net = network.DNNNet(phase="test", num_classes=10, hidden_nums=128)

    _, pred = dnn_net.loss(labels=labels, input_data=input_layer)

    saver = tf.train.Saver()

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=sess_config)

    # start training
    with sess.as_default():
        if weight_path is None:
            raise ValueError("weight path doesn't configured")

        logger.info("Restore model from {:s}".format(weight_path))
        saver.restore(sess=sess, save_path=weight_path)
        test_acc_list = []

        # omit the last few data that less than a batch size
        batch_num = int(np.floor(test_labels.size) / batch_size)

        for i in range(batch_num):
            data_batch = test_data[i * batch_size:(i + 1) * batch_size]
            labels_batch = test_labels[i * batch_size:(i + 1) * batch_size]
            pred_label = sess.run(pred, feed_dict={input_layer: data_batch,
                                                   labels: labels_batch})
            test_acc = compute_acc(preds=pred_label, labels=labels_batch)
            test_acc_list.append(test_acc)
        acc = sum(test_acc_list) / len(test_acc_list) * 100
        logger.info("test acc: {:.2f}%%".format(acc))


if __name__ == '__main__':
    init_opt = my_parser()
    test_type = init_opt.type.upper()
    if test_type == "CNN":
        cnn_test(batch_size=init_opt.batch_size, weight_path=init_opt.weight_path)
    elif test_type == "DNN":
        dnn_test(batch_size=init_opt.batch_size, weight_path=init_opt.weight_path)
    else:
        raise ValueError("{:s} type is not supported!".format(test_type))

    logger.info("done!")
