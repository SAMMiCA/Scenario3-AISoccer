import tensorflow as tf
import tensorflow.contrib.layers as layers

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    # This model takes as input an observation and returns values of all actions
    print("Reusing MLP_MODEL: {}".format(reuse))
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def lstm_fc_model(input_ph, num_outputs, scope, reuse=False, num_units=64):
    print("Reusing LSTM_FC_MODEL: {}".format(reuse))
    with tf.variable_scope(scope, reuse=reuse):
        input_, c_, h_ = input_ph[:,:,:-2*num_units], input_ph[:,:,-2*num_units:-1*num_units], input_ph[:,:,-1*num_units:]
        out = input_
        out = layers.fully_connected(out, num_outputs=int(input_.shape[-1]), activation_fn=tf.nn.relu)
        c_, h_ = tf.squeeze(c_, [1]), tf.squeeze(h_, [1])
        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
        state = tf.contrib.rnn.LSTMStateTuple(c_,h_)
        out, state = tf.nn.dynamic_rnn(cell, out, initial_state=state)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        c_, h_ = tf.expand_dims(state.c, axis=1), tf.expand_dims(state.h, axis=1) # ensure same shape as input state
        state = tf.contrib.rnn.LSTMStateTuple(c_,h_)
        return out, state
