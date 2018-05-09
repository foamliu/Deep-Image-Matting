# dependency: pip install pydot & brew install graphviz
from new_start import autoencoder
from keras.utils import plot_model


if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 3
    model = autoencoder(img_rows, img_cols, channel)
    plot_model(model, to_file='model.svg', show_layer_names=True, show_shapes=True)
