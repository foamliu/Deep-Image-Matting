import keras.backend as K
from keras.layers import Input, Conv2D, Add, Lambda
from keras.models import Model
from keras.utils import plot_model

img_rows, img_cols = 320, 320
channel = 4
feature_size = 64
kernel = 3
num_layers = 32
scaling_factor = .1


def res_block(x, channels=64, scale=1.0):
    tmp = Conv2D(channels, (kernel, kernel), activation='relu', padding='same')(x)
    tmp = Conv2D(channels, (kernel, kernel), padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)
    return Add()([x, tmp])


def build_model():
    input_tensor = Input(shape=(img_rows, img_cols, channel))

    # One convolution before res blocks and to convert to required feature depth
    x = Conv2D(feature_size, (kernel, kernel), activation='relu', padding='same', name='conv1')(input_tensor)

    # Store the output of the first convolution to add later
    conv_1 = x

    """
    This creates `num_layers` number of resBlocks
    a resBlock is defined in the paper as
    (excuse the ugly ASCII graph)
    x
    |\
    | \
    |  conv2d
    |  relu
    |  conv2d
    | /
    |/
    + (addition here)
    |
    result
    """

    """
    Doing scaling here as mentioned in the paper:
    `we found that increasing the number of feature
    maps above a certain level would make the training procedure
    numerically unstable. A similar phenomenon was
    reported by Szegedy et al. We resolve this issue by
    adopting the residual scaling with factor 0.1. In each
    residual block, constant scaling layers are placed after the
    last convolution layers. These modules stabilize the training
    procedure greatly when using a large number of filters.
    In the test phase, this layer can be integrated into the previous
    convolution layer for the computational efficiency.'
    """

    # Add the residual blocks to the model
    for i in range(num_layers):
        x = res_block(x, feature_size, scale=scaling_factor)

    x = Conv2D(feature_size, (kernel, kernel), padding='same')(x)
    x = Add()([x, conv_1])
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='pred')(x)

    outputs = x

    model = Model(inputs=input_tensor, outputs=outputs, name="ResNet-Matting")
    return model


if __name__ == '__main__':
    m = build_model()
    print(m.summary())
    plot_model(m, to_file='encoder_decoder.svg', show_layer_names=True, show_shapes=True)
    K.clear_session()
