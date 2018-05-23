from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape, Concatenate, Lambda, Multiply


class Unpooling(Layer):

    def __init__(self, **kwargs):
        super(Unpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Unpooling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs[:, 1]
        # print('x.shape: ' + str(K.int_shape(x)))
        bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]),
                           output_shape=K.int_shape(x)[1:])(inputs)
        # print('bool_mask.shape: ' + str(K.int_shape(bool_mask)))
        mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)
        # print('mask.shape: ' + str(K.int_shape(mask)))
        x = Multiply()([mask, x])
        # print('x.shape: ' + str(K.int_shape(x)))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3], input_shape[4]
