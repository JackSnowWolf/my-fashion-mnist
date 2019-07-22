import os
import tensorflow as tf
from optparse import OptionParser
import time
import numpy as np
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
    parser.add_option("-t", "--type", action="store", dest="type",
                      default="CNN", type="string",
                      help="set cnn or dnn")
    parser.add_option("--lr", "--learning_rate", action="store",
                      dest="learning_rate",
                      type="float", default=0.5,
                      help="set learning_rate")
    parser.add_option("--batch_size", action="store", dest="batch_size",
                      type="int", default=1,
                      help="set batch size")
    parser.add_option("-w", "--weight", action="store", dest="weight_path",
                      type="string",
                      default=None,
                      help="path to pretrain weight or previous weight")
    parser.add_option("--tboard", action="store", dest="tboard",
                      type="string", default="tboard",
                      help="set tensor board log directory")
    parser.add_option("-n", "--steps", action="store", dest="steps",
                      default=1000, type="int", help="set train steps")
    parser.add_option("-s", "--save_path", action="store", dest="save_path",
                      type="string", default="model",
                      help="set model save path")

    options, _ = parser.parse_args()
    return options


def cnn_train(lr, batch_size, weight_path, train_epochs, tboard_dir, save_path):
    # load data
    train_data, train_labels, val_data, val_labels = data_loader.data_load("data/fashion", kind="train", partition=0.17)
    assert len(train_data) == len(train_labels)

    # set configuration

    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name="input")
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="labels")
    cnn_net = network.CNNNet(phase="train", num_classes=10)

    loss, pred = cnn_net.loss(labels=labels, input_data=input_layer)

    # set learning rate
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # # MomentumOptimizer
    # learning_rate = tf.train.exponential_decay(learning_rate=lr, global_step=global_step,
    #                                            decay_steps=300, decay_rate=0.5, staircase=True)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # AdamOptimizer
    learning_rate = tf.Variable(lr, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grad = optimizer.compute_gradients(loss=loss)

    apply_grad_op = optimizer.apply_gradients(grad, global_step=global_step)

    # set tensorflow summary
    tboard_save_path = os.path.join(tboard_dir, "cnn")
    os.makedirs(tboard_save_path, exist_ok=True)
    summary = tf.summary.FileWriter(tboard_save_path)

    train_loss_scalar = tf.summary.scalar(name="train_loss", tensor=loss)
    learning_rate_scalar = tf.summary.scalar(name="learning_rate", tensor=learning_rate)
    train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)
    # train_merge_summary_op = tf.summary.merge([train_loss_scalar, learning_rate_scalar], train_summary_op_updates)
    train_merge_summary_op = tf.summary.merge_all()

    # set saver
    os.makedirs(os.path.join(save_path, "cnn"), exist_ok=True)

    saver = tf.train.Saver(max_to_keep=20)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=sess_config)
    summary.add_graph(sess.graph)

    # start training
    with sess.as_default():
        epoch = 0
        if weight_path is None:
            logger.info("Training from scratch")
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info("Restore model from {:s}".format(weight_path))
            saver.restore(sess=sess, save_path=weight_path)
        train_loss_list = []
        train_acc_list = []
        while epoch < train_epochs:
            epoch += 1
            batch_index = np.random.choice(np.arange(train_labels.size), size=batch_size, replace=False)
            labels_batch = train_labels[batch_index]
            data_batch = train_data[batch_index]
            _, train_loss, pred_label, train_merge_summary_value = sess.run(
                [apply_grad_op, loss, pred, train_merge_summary_op],
                feed_dict={input_layer: data_batch,
                           labels: labels_batch})
            acc = compute_acc(preds=pred_label, labels=labels_batch) * 100
            train_loss_list.append(train_loss)
            train_acc_list.append(acc)
            if epoch % 10 == 0:
                acc = sum(train_acc_list) / len(train_acc_list)
                train_loss = sum(train_loss_list) / len(train_loss_list)
                train_acc_list = []
                train_loss_list = []
                logger.info("epoch: {:d}\ttrain loss: {:.3f}\ttrain acc: {:.2f}%%".format(epoch, train_loss, acc))
                summary.add_summary(summary=train_merge_summary_value, global_step=epoch)
            if epoch % 100 == 0:
                # validation
                val_acc_list = []
                val_loss_list = []
                batch_num = int(np.floor(val_labels.size / batch_size))
                for i in range(batch_num):
                    data_batch = val_data[i * batch_size:(i + 1) * batch_size]
                    labels_batch = val_labels[i * batch_size:(i + 1) * batch_size]
                    val_loss, pred_label = sess.run([loss, pred], feed_dict={input_layer: data_batch,
                                                                             labels: labels_batch})
                    val_acc = compute_acc(preds=pred_label, labels=labels_batch)
                    val_acc_list.append(val_acc)
                    val_loss_list.append(val_loss)
                acc = sum(val_acc_list) / len(val_acc_list) * 100
                val_loss = sum(val_loss_list) / len(val_loss_list)
                logger.info("epoch: {:d}\tval loss: {:.3f}\tval acc: {:.2f}%%".format(epoch, val_loss, acc))

                model_name = 'cnn_{:s}_{:06d}.ckpt'.format(str(train_start_time), epoch)
                model_save_path = os.path.join(save_path, "cnn", model_name)
                saver.save(sess=sess, save_path=model_save_path)
        model_name = 'cnn_{:s}_{:06d}.ckpt'.format(str(train_start_time), epoch)
        model_save_path = os.path.join(save_path, "cnn", model_name)
        saver.save(sess=sess, save_path=model_save_path)


def dnn_train(lr, batch_size, weight_path, train_epochs, tboard_dir, save_path):
    # load data
    train_data, train_labels, val_data, val_labels = data_loader.data_load("data/fashion", kind="train", partition=0.17)
    assert len(train_data) == len(train_labels)

    # set configuration

    # construct network structure
    input_layer = tf.placeholder(dtype=tf.float32, shape=[batch_size, 784], name="input")
    labels = tf.placeholder(dtype=tf.int32, shape=[batch_size], name="labels")
    dnn_net = network.DNNNet(phase="train", num_classes=10, hidden_nums=128)

    loss, pred = dnn_net.loss(labels=labels, input_data=input_layer)

    # set learning rate
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # # MomentumOptimizer
    # learning_rate = tf.train.exponential_decay(learning_rate=lr, global_step=global_step,
    #                                            decay_steps=300, decay_rate=0.5, staircase=True)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

    # AdamOptimizer
    learning_rate = tf.Variable(lr, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grad = optimizer.compute_gradients(loss=loss)

    apply_grad_op = optimizer.apply_gradients(grad, global_step=global_step)

    # set tensorflow summary
    tboard_save_path = os.path.join(tboard_dir, "dnn")
    os.makedirs(tboard_save_path, exist_ok=True)
    summary = tf.summary.FileWriter(tboard_save_path)

    train_loss_scalar = tf.summary.scalar(name="train_loss", tensor=loss)
    learning_rate_scalar = tf.summary.scalar(name="learning_rate", tensor=learning_rate)
    train_summary_op_updates = tf.get_collection(tf.GraphKeys.SUMMARIES)
    # train_merge_summary_op = tf.summary.merge([train_loss_scalar, learning_rate_scalar], train_summary_op_updates)
    train_merge_summary_op = tf.summary.merge_all()

    # set saver
    os.makedirs(os.path.join(save_path, "dnn"), exist_ok=True)

    saver = tf.train.Saver(max_to_keep=20)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    # set sess config
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2

    sess = tf.Session(config=sess_config)
    summary.add_graph(sess.graph)

    # start training
    with sess.as_default():
        epoch = 0
        if weight_path is None:
            logger.info("Training from scratch")
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info("Restore model from {:s}".format(weight_path))
            saver.restore(sess=sess, save_path=weight_path)
        train_loss_list = []
        train_acc_list = []
        while epoch < train_epochs:
            epoch += 1
            batch_index = np.random.choice(np.arange(train_labels.size), size=batch_size, replace=False)
            labels_batch = train_labels[batch_index]
            data_batch = train_data[batch_index]
            _, train_loss, pred_label, train_merge_summary_value = sess.run(
                [apply_grad_op, loss, pred, train_merge_summary_op],
                feed_dict={input_layer: data_batch,
                           labels: labels_batch})
            acc = compute_acc(preds=pred_label, labels=labels_batch) * 100
            train_loss_list.append(train_loss)
            train_acc_list.append(acc)
            if epoch % 10 == 0:
                acc = sum(train_acc_list) / len(train_acc_list)
                train_loss = sum(train_loss_list) / len(train_loss_list)
                train_acc_list = []
                train_loss_list = []
                logger.info("epoch: {:d}\ttrain loss: {:.3f}\ttrain acc: {:.2f}%%".format(epoch, train_loss, acc))
                summary.add_summary(summary=train_merge_summary_value, global_step=epoch)
            if epoch % 100 == 0:
                # validation
                val_acc_list = []
                val_loss_list = []
                batch_num = int(np.floor(val_labels.size / batch_size))
                for i in range(batch_num):
                    data_batch = val_data[i * batch_size:(i + 1) * batch_size]
                    labels_batch = val_labels[i * batch_size:(i + 1) * batch_size]
                    val_loss, pred_label = sess.run([loss, pred], feed_dict={input_layer: data_batch,
                                                                             labels: labels_batch})
                    val_acc = compute_acc(preds=pred_label, labels=labels_batch)
                    val_acc_list.append(val_acc)
                    val_loss_list.append(val_loss)
                acc = sum(val_acc_list) / len(val_acc_list) * 100
                val_loss = sum(val_loss_list) / len(val_loss_list)
                logger.info("epoch: {:d}\tval loss: {:.3f}\tval acc: {:.2f}%%".format(epoch, val_loss, acc))
                model_name = 'dnn_{:s}_{:06d}.ckpt'.format(str(train_start_time), epoch)
                model_save_path = os.path.join(save_path, "dnn", model_name)
                saver.save(sess=sess, save_path=model_save_path)
        model_name = 'dnn_{:s}_{:06d}.ckpt'.format(str(train_start_time), epoch)
        model_save_path = os.path.join(save_path, "dnn", model_name)
        saver.save(sess=sess, save_path=model_save_path)


if __name__ == '__main__':
    init_opt = my_parser()
    train_type = init_opt.type.upper()
    if train_type == "CNN":
        cnn_train(lr=init_opt.learning_rate, batch_size=init_opt.batch_size, weight_path=init_opt.weight_path,
                  train_epochs=init_opt.steps, tboard_dir=init_opt.tboard, save_path=init_opt.save_path)
    elif train_type == "DNN":
        dnn_train(lr=init_opt.learning_rate, batch_size=init_opt.batch_size, weight_path=init_opt.weight_path,
                  train_epochs=init_opt.steps, tboard_dir=init_opt.tboard, save_path=init_opt.save_path)
    else:
        raise ValueError("{:s} type is not supported!".format(train_type))

    logger.info("done!")
