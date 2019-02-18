from src.models.RNN import *

class LSTMRegression(RNNRegression):
    def __init__(self, n_neurons=120, n_layers=1, learning_rate=0.01, fit_range=50,
                       steps_ahead=30,
                       optimizer_class=tf.train.AdamOptimizer):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.fit_range = fit_range
        self.steps_ahead = steps_ahead
        self.optimizer_class = optimizer_class
        self.n_inputs = 1
        self.n_outputs = 1
        self._session = None

    def _build_graph(self):
        X = tf.placeholder(tf.float32, [None, self.fit_range, self.n_inputs])
        y = tf.placeholder(tf.float32, [None, self.fit_range, self.n_outputs])
        keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Defining LSTM connections
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.n_neurons, use_peepholes=True)
        #                                    activation=tf.nn.elu)
        #cell_drop = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=keep_prob)
        lstm_outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)

        stacked_lstm_outputs = tf.reshape(lstm_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_lstm_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, self.fit_range, self.n_outputs])

        # Defining operations (loss, optimizer)
        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = self.optimizer_class(learning_rate=self.learning_rate, beta1=0.93)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y, self._keep_prob = X, y, keep_prob
        self._outputs = outputs
        self._loss = loss
        self._training_op = training_op
        self._init, self._saver = init, saver
