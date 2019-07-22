import tensorflow as tf


class BaseNet:
    def __init__(self, phase="train", num_classes=10):
        self.phase = phase.upper()
        self.num_classes = num_classes

    def is_train(self):
        if self.phase == "TRAIN":
            return True
        elif self.phase == "TEST":
            return False
        else:
            raise ValueError("Not a valid phase")

    def inference(self, input_data):
        raise NotImplementedError

    def loss(self, labels, input_data):
        raise NotImplementedError

    def forward(self, input_data):
        raise NotImplementedError


class CNNNet(BaseNet):
    def __init__(self, phase="train", num_classes=10):
        super(CNNNet, self).__init__(phase, num_classes)

    def inference(self, input_data):
        """
        inference
        :param input_data:
        :return: return prediction label and raw output
        """

        output_layer = self.forward(input_data)
        pred = tf.argmax(tf.nn.softmax(output_layer), axis=1, name="pred")
        return pred, output_layer

    def loss(self, labels, input_data):
        """
        compute and loss and do inference
        :param labels: ground truth lable
        :param input_data: input data [batch x num_cells]
        :return: loss and prediction label
        """
        with tf.variable_scope(name_or_scope="cnn"):
            pred, out = self.inference(input_data)
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, out), name="loss")
            return loss, pred

    def forward(self, input_data):
        """
        forward process
        :param input_data: DNN input_data [batch x input_cells]
        :return: output result [batch x num_classes]
        """

        # reshape input data input NHWC format
        [batch_num, _] = input_data.get_shape().as_list()
        reshaped = self.reshape(input_data)

        # conv1
        w1 = tf.get_variable('w1', [3, 3, 1, 32], initializer=tf.contrib.layers.variance_scaling_initializer(),
                             trainable=self.is_train())
        conv1 = tf.nn.conv2d(reshaped, w1, [1, 1, 1, 1], "SAME", name="conv1")
        bn1 = tf.contrib.layers.batch_norm(conv1,  scope="bn1", trainable=self.is_train())
        relu1 = tf.nn.relu(bn1, name="relu1")
        maxpool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="maxpool1")

        # conv2
        w2 = tf.get_variable('w2', [3, 3, 32, 64], initializer=tf.contrib.layers.variance_scaling_initializer(),
                             trainable=self.is_train())
        conv2 = tf.nn.conv2d(maxpool1, w2, [1, 1, 1, 1], "SAME", name="conv2")
        bn2 = tf.contrib.layers.batch_norm(conv2,  scope="bn2", trainable=self.is_train())
        relu2 = tf.nn.relu(bn2, name="relu2")
        maxpool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="maxpool2")

        # fully connection
        before_fc = tf.reshape(maxpool2, [-1, 7 * 7 * 64])

        fc1 = tf.layers.dense(before_fc, 1024, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              name="fc1", trainable=self.is_train())
        fc2 = tf.layers.dense(fc1, self.num_classes,
                              kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                              name="fc2",  trainable=self.is_train())
        return fc2

    def reshape(self, input_data):
        """
        reshape input_data into NHWC format
        :param input_data:
        :return: reshaped input_data, in NHWC format
        """
        [batch_num, _] = input_data.get_shape().as_list()
        reshaped = tf.reshape(input_data, shape=[batch_num, 28, 28, 1], name="reshaped")
        return reshaped


class DNNNet(BaseNet):
    def __init__(self, phase="train", num_classes=10, hidden_nums=128):
        super(DNNNet, self).__init__(phase, num_classes)
        self.hidden_nums = hidden_nums

    def inference(self, input_data):
        """
        inference
        :param input_data:
        :return: return prediction label and raw output
        """
        output_layer = self.forward(input_data)
        pred = tf.argmax(tf.nn.softmax(output_layer), axis=1, name="pred")
        return pred, output_layer

    def loss(self, labels, input_data):
        """
        compute and loss and do inference
        :param labels: ground truth lable
        :param input_data: input data [batch x num_cells]
        :return: loss and prediction label
        """
        with tf.variable_scope(name_or_scope="dnn"):
            pred, out = self.inference(input_data)
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels, out), name="loss")
            return loss, pred

    def forward(self, input_data):
        """
        forward process
        :param input_data: DNN input_data [batch x input_cells]
        :return: output result [batch x num_classes]
        """

        [_, input_cells] = input_data.get_shape().as_list()
        w_0 = tf.get_variable(name='w_0',
                              shape=[input_cells, self.hidden_nums],
                              initializer=tf.truncated_normal_initializer(stddev=0.02),
                              trainable=self.is_train())
        hidden_layer = tf.matmul(input_data, w_0, name="hidden_layer")

        relu_1 = tf.nn.relu(hidden_layer, name="relu_1")
        w_1 = tf.get_variable(name="w_1",
                              shape=[self.hidden_nums, self.num_classes],
                              initializer=tf.truncated_normal_initializer(stddev=0.02),
                              trainable=self.is_train())
        output_layer = tf.matmul(relu_1, w_1, name="output_layer")
        # relu_2 = tf.nn.relu(output_layer, name="relu_2")
        return output_layer
