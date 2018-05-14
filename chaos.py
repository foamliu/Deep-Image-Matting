import cv2 as cv
import numpy as np

from model import create_model

# python test.py -i "images/image.png" -t "images/trimap.png"
if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    model_weights_path = 'models/model.48-0.03.hdf5'
    model = create_model(img_rows, img_cols, channel)
    model.load_weights(model_weights_path)
    print(model.summary())

    image_path = 'images/image_B.png'
    trimap_path = 'images/trimap_B.png'
    back_path = 'images/back_B.png'
    back_path = 'images/back_B.png'
    back_path = 'images/back_B.png'

    print('Start processing image: {}'.format(image_path))

    x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
    bgr_img = cv.imread(image_path)
    trimap = cv.imread(trimap_path, 0)
    back = cv.imread(back_path)

    x, y = 300, 200
    bgr_img = bgr_img[y:y + 320, x:x + 320, :]
    trimap = trimap[y:y + 320, x:x + 320]
    back = back[y:y + 320, x:x + 320, :]

    x_test = np.empty((1, 320, 320, 4), dtype=np.float32)
    x_test[0, :, :, 0:3] = bgr_img / 255.
    x_test[0, :, :, 3] = trimap / 255.

    alpha = model.predict(x_test)

    merge = alpha[0] * bgr_img + (1 - alpha[0]) * back
    print(merge.shape)
    merge = np.reshape(merge, (320, 320, 3))

    merge = merge.astype(np.uint8)

    cv.imshow('merge_s', merge)
    cv.imwrite('images/image_s.png', bgr_img)
    cv.imwrite('images/trimap_s.png', trimap)
    cv.imwrite('images/back_s.png', back)
    cv.imwrite('images/out_s.png', merge)
    cv.waitKey(0)
    cv.destroyAllWindows()
