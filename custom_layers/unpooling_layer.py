from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Reshape, Concatenate, Lambda, Multiply


class Unpooling(Layer):

    def __init__(self, orig, the_shape, **kwargs):
        self.orig = orig
        self.the_shape = the_shape
        super(Unpooling, self).__init__(**kwargs)

    def call(self, x):
        # here we're going to reshape the data for a concatenation:
        # xReshaped and origReshaped are now split branches
        shape = list(self.the_shape)
        shape.insert(0, 1)
        shape = tuple(shape)
        xReshaped = Reshape(shape)(x)
        origReshaped = Reshape(shape)(self.orig)

        # concatenation - here, you unite both branches again
        # normally you don't need to reshape or use the axis var,
        # but here we want to keep track of what was x and what was orig.
        together = Concatenate(axis=1)([origReshaped, xReshaped])

        bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]),
                           output_shape=self.the_shape)(together)
        mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)

        x = Multiply()([mask, x])
        return x
