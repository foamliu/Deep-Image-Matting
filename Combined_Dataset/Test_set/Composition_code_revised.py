import math
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm

##############################################################
# Set your paths here

# path to provided foreground images
fg_path = 'data/fg_test/'

# path to provided alpha mattes
a_path = 'data/mask_test/'

# Path to background images (MSCOCO)
bg_path = 'data/bg_test/'

# Path to folder where you want the composited images to go
out_path = 'data/merged_test/'


##############################################################

def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg = np.array(bg[0:h, 0:w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.
    comp = alpha * fg + (1 - alpha) * bg
    comp = comp.astype(np.uint8)
    return comp


def process(im_name, bg_name, fcount, bcount):
    im = cv.imread(fg_path + im_name)
    a = cv.imread(a_path + im_name, 0)
    h, w = im.shape[:2]
    bg = cv.imread(bg_path + bg_name)
    bh, bw = bg.shape[:2]
    wratio = w / bw
    hratio = h / bh
    ratio = wratio if wratio > hratio else hratio
    if ratio > 1:
        bg = cv.resize(src=bg, dsize=(math.ceil(bw * ratio), math.ceil(bh * ratio)), interpolation=cv.INTER_CUBIC)

    out = composite4(im, bg, a, w, h)
    filename = out_path + str(fcount) + '_' + str(bcount) + '.png'
    cv.imwrite(filename, out)


def do_composite_test():
    num_bgs = 20

    with open('Combined_Dataset/Test_set/test_bg_names.txt') as f:
        bg_files = f.read().splitlines()
    with open('Combined_Dataset/Test_set/test_fg_names.txt') as f:
        fg_files = f.read().splitlines()

    # a_files = os.listdir(a_path)
    num_samples = len(fg_files) * num_bgs

    # pb = ProgressBar(total=100, prefix='Compose test images', suffix='', decimals=3, length=50, fill='=')
    start = time.time()
    bcount = 0
    for fcount in tqdm(range(len(fg_files))):
        im_name = fg_files[fcount]

        for i in range(num_bgs):
            bg_name = bg_files[bcount]
            process(im_name, bg_name, fcount, bcount)
            bcount += 1

            # pb.print_progress_bar(bcount * 100.0 / num_samples)

    end = time.time()
    elapsed = end - start
    print('elapsed: {} seconds'.format(elapsed))
