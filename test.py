import tensorflow as tf
import numpy as np
from scipy import misc
from matting import generate_trimap
import argparse
import sys

g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])


def main(args):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model'))
        image_batch = tf.get_collection('image_batch')[0]
        GT_trimap = tf.get_collection('GT_trimap')[0]
        pred_mattes = tf.get_collection('pred_mattes')[0]

        rgb = misc.imread(args.rgb)
        misc.imshow(rgb)
        alpha = misc.imread(args.alpha, 'L')
        misc.imshow(alpha)
        trimap = generate_trimap(np.expand_dims(np.copy(alpha), 2), np.expand_dims(alpha, 2))[:, :, 0]
        misc.imshow(trimap)
        origin_shape = alpha.shape
        rgb = np.expand_dims(misc.imresize(rgb.astype(np.uint8), [320, 320, 3]).astype(np.float32) - g_mean, 0)
        trimap = np.expand_dims(
            np.expand_dims(misc.imresize(trimap.astype(np.uint8), [320, 320], interp='nearest').astype(np.float32), 2),
            0)

        feed_dict = {image_batch: rgb, GT_trimap: trimap}
        pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)
        final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)
        misc.imshow(final_alpha)
        misc.imsave('./alpha.png', final_alpha)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--alpha', type=str,
                        help='input alpha')
    parser.add_argument('--rgb', type=str,
                        help='input rgb')
    parser.add_argument('--gpu_fraction', type=float,
                        help='how much gpu is needed, usually 4G is enough', default=0.4)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
