from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape, Concatenate, Lambda, Multiply


class Unpooling(Layer):

    def __init__(self, orig, **kwargs):
        self.orig = orig
        super(Unpooling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Unpooling, self).build(input_shape)

    def call(self, x, **kwargs):
        # here we're going to reshape the data for a concatenation:
        # xReshaped and origReshaped are now split branches
        the_shape = K.int_shape(self.orig)
        shape = (1, the_shape[1], the_shape[2], the_shape[3])
        origReshaped = Reshape(shape)(self.orig)
        xReshaped = Reshape(shape)(x)

        # concatenation - here, you unite both branches again
        # normally you don't need to reshape or use the axis var,
        # but here we want to keep track of what was x and what was orig.
        together = Concatenate(axis=1)([origReshaped, xReshaped])

        bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]),
                           output_shape=(the_shape[1], the_shape[2], the_shape[3]))(together)
        mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)

        x = Multiply()([mask, x])
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
