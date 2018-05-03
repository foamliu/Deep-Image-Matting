import os
import numpy as np
import cv2 as cv
import random
import keras.backend as K
from utils import  matting_loss

i1 = random.randint(1, 8041)
i2 = random.randint(1, 8041)

image1 = os.path.join('data/test', '%05d.jpg' % (i1 + 1))
image2 = os.path.join('data/test', '%05d.jpg' % (i2 + 1))

bgr_img1 = cv.imread(image1)
y_true = cv.cvtColor(bgr_img1, cv.COLOR_BGR2GRAY)
y_true = np.array(y_true, np.float32)
bgr_img2 = cv.imread(image2)
y_pred = cv.cvtColor(bgr_img2, cv.COLOR_BGR2GRAY)
y_pred = np.array(y_pred, np.float32)


loss = matting_loss(y_true, y_pred)
ret = K.eval(loss)
print(ret)
