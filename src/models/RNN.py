import numpy as np
from random import sample, seed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
import tensorflow as tf

def splitData(series, steps_ahead, fit_range, ratio = 0.7):
    '''
    Creates X_train, y_train, X_test, y_test, X_total
    '''
    seed(10)
    lenSeries = len(series)-steps_ahead-fit_range
    indices = sample(range(lenSeries),lenSeries)
    train_idx, test_idx = indices[:int(ratio*lenSeries)], indices[int(ratio*lenSeries):]
    series = np.asarray(series['Close'])
    X_total = []
    for t in range(len(series)-fit_range):
        X_total.append( series[t:(t+fit_range)] )
    X_train = []
    y_train = []
    for t in train_idx:
        X_train.append( series[t:(t+fit_range)] )
        y_train.append( series[(t+steps_ahead):(t+fit_range+steps_ahead)] )
    X_test = []
    y_test = []
    for t in test_idx:
        X_test.append( series[t:(t+fit_range)] )
        y_test.append( series[(t+steps_ahead):(t+fit_range+steps_ahead)] )
    return np.array(X_train).reshape(-1, fit_range, 1), np.array(y_train).reshape(-1, fit_range, 1), np.array(X_test).reshape(-1, fit_range, 1), np.array(y_test).reshape(-1, fit_range, 1), np.array(X_total).reshape(-1, fit_range, 1)

class RNNRegression(BaseEstimator, RegressorMixin):
    def __init__(self, n_neurons=120, learning_rate=0.01, fit_range=50, steps_ahead=30,
                       optimizer_class=tf.train.AdamOptimizer, activation=tf.nn.elu):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.fit_range = fit_range
        self.steps_ahead = steps_ahead
        self.optimizer_class = optimizer_class
        self.activation = activation
        self.n_inputs = 1
        self.n_outputs = 1
        self._session = None

    def _build_graph(self):
        X = tf.placeholder(tf.float32, [None, self.fit_range, self.n_inputs])
        y = tf.placeholder(tf.float32, [None, self.fit_range, self.n_outputs])
        keep_prob = tf.placeholder_with_default(1.0, shape=())

        # Defining RNN connections
        cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons, activation=self.activation)
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
        rnn_outputs, states = tf.nn.dynamic_rnn(cell_drop, X, dtype=tf.float32)

        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, self.fit_range, self.n_outputs])

        # Defining operations (loss, optimizer)
        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = self.optimizer_class(learning_rate=self.learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Make the important operations available easily through instance variables
        self._X, self._y, self._keep_prob = X, y, keep_prob
        self._outputs = outputs
        self._loss = loss
        self._training_op = training_op
        self._init, self._saver = init, saver

    def close_session(self):
        if self._session:
            self._session.close()

    def _get_model_params(self):
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        gvar_names = list(model_params.keys())
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names}
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)

    def fit(self, X, max_iterations=1500, keep_prob=0.5):
        '''
        Trains RNN and saves model. Prints out RMSE on the test data if available
        '''
        X_train, y_train, X_test, y_test, X_total = splitData(X, self.steps_ahead, self.fit_range, ratio = 0.7)

        self.close_session()

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()

        # needed in case of early stopping
        best_rmse = np.infty
        iterations_without_progress = 0
        if keep_prob == 1:
            max_iterations_without_progress = 600
        else:
            max_iterations_without_progress = 1000
        best_params = None

        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            self._init.run()
            for iteration in range(max_iterations):
                sess.run(self._training_op, feed_dict={self._X: X_train, self._y: y_train, self._keep_prob: keep_prob})
                if iteration % 20 == 0:
                    rmse = np.sqrt(self._loss.eval(feed_dict={self._X: X_test, self._y: y_test, self._keep_prob: 1}))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_params = self._get_model_params()
                        iterations_without_progress = 0
                    else:
                        iterations_without_progress += 20
                    print("Iteration {0} - model RMSE:{1:.3f} - best RMSE:{2:.3f}".format(iteration, rmse, best_rmse))
                    if iterations_without_progress > max_iterations_without_progress:
                        print("Early stopping!")
                        break
            if best_params:
                self._restore_model_params(best_params)
        return self

    def predict(self, X):
        X_train, y_train, X_test, y_test, X_total = splitData(X, self.steps_ahead, self.fit_range, ratio = 0.7)
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._outputs.eval(feed_dict={self._X: X_total, self._keep_prob: 1})

    def score(self, X):
        X_train, y_train, X_test, y_test, X_total = splitData(X, self.steps_ahead, self.fit_range, ratio = 0.7)
        if not self._session:
            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)
        with self._session.as_default() as sess:
            rmse = np.sqrt(self._loss.eval(feed_dict={self._X: X_test, self._y: y_test, self._keep_prob: 1}))
            return(-rmse)

    def rolling_estimate(self, X):
        X_train, y_train, X_test, y_test, X_total = splitData(X, self.steps_ahead, self.fit_range, ratio = 0.7)
        res = self.predict(X)
        pred = []
        for t in range(len(X_total)):
            pred.append(res[t][-1][0])
        return(pred)

    def save(self, path="../models/RNNmodel"):
        self._saver.save(self._session, path)

    def restore_model(self, path="../models/RNNmodel"):
        self.close_session()
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            try:
                self._saver.restore(sess, path)
            except:
                print('Does the file exist? did you set the hyper-parameters the same as in the save?')
        return self
