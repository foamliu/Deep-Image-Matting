# -*- coding: utf-8 -*-

import os
import shutil
import zipfile
import tarfile

from Combined_Dataset.Training_set.Composition_code_revised import do_composite
from Combined_Dataset.Test_set.Composition_code_revised import do_composite_test

if __name__ == '__main__':
    # path to provided foreground images
    fg_path = 'fg/'
    # path to provided alpha mattes
    a_path = 'mask/'
    # Path to background images (MSCOCO)
    bg_path = 'bg/'
    # Path to folder where you want the composited images to go
    out_path = 'merged/'

    train_folder = 'Combined_Dataset/Training_set/'

    # if not os.path.exists('Combined_Dataset'):
    zip_file = 'Adobe_Deep_Matting_Dataset.zip'
    print('Extracting {}...'.format(zip_file))

    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall('.')
    zip_ref.close()

    if not os.path.exists(bg_path):
        zip_file = 'train2014.zip'
        print('Extracting {}...'.format(zip_file))

        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall('.')
        zip_ref.close()

        training_bg_names = []
        with open(os.path.join(train_folder, 'training_bg_names.txt')) as f:
            training_bg_names = f.read().splitlines()

        os.makedirs(bg_path)
        for bg_name in training_bg_names:
            src_path = os.path.join('train2014', bg_name)
            dest_path = os.path.join(bg_path, bg_name)
            shutil.move(src_path, dest_path)

    if not os.path.exists(fg_path):
        os.makedirs(fg_path)

    for old_folder in [train_folder + 'Adobe-licensed images/fg', train_folder + 'Other/fg']:
        fg_files = os.listdir(old_folder)
        for fg_file in fg_files:
            src_path = os.path.join(old_folder, fg_file)
            dest_path = os.path.join(fg_path, fg_file)
            shutil.move(src_path, dest_path)

    if not os.path.exists(a_path):
        os.makedirs(a_path)

    for old_folder in [train_folder + 'Adobe-licensed images/alpha', train_folder + 'Other/alpha']:
        a_files = os.listdir(old_folder)
        for a_file in a_files:
            src_path = os.path.join(old_folder, a_file)
            dest_path = os.path.join(a_path, a_file)
            shutil.move(src_path, dest_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # do_composite()

    # path to provided foreground images
    fg_test_path = 'fg_test/'
    # path to provided alpha mattes
    a_test_path = 'mask_test/'
    # Path to background images (PASCAL VOC)
    bg_test_path = 'bg_test/'
    # Path to folder where you want the composited images to go
    out_test_path = 'merged_test/'

    # test data gen
    test_folder = 'Combined_Dataset/Test_set/'

    if not os.path.exists(bg_test_path):
        os.makedirs(bg_test_path)

    tar_file = 'VOCtrainval_14-Jul-2008.tar'
    print('Extracting {}...'.format(tar_file))

    tar = tarfile.open(tar_file)
    tar.extractall()
    tar.close()

    tar_file = 'VOC2008test.tar'
    print('Extracting {}...'.format(tar_file))

    tar = tarfile.open(tar_file)
    tar.extractall()
    tar.close()

    test_bg_names = []
    with open(os.path.join(test_folder, 'test_bg_names.txt')) as f:
        test_bg_names = f.read().splitlines()

    for bg_name in test_bg_names:
        tokens = bg_name.split('_')
        src_path = os.path.join('VOCdevkit/VOC2008/JPEGImages', bg_name)
        dest_path = os.path.join(bg_test_path, bg_name)
        shutil.move(src_path, dest_path)

    if not os.path.exists(fg_test_path):
        os.makedirs(fg_test_path)

    for old_folder in [test_folder + 'Adobe-licensed images/fg']:
        fg_files = os.listdir(old_folder)
        for fg_file in fg_files:
            src_path = os.path.join(old_folder, fg_file)
            dest_path = os.path.join(fg_test_path, fg_file)
            shutil.move(src_path, dest_path)

    if not os.path.exists(a_test_path):
        os.makedirs(a_test_path)

    for old_folder in [test_folder + 'Adobe-licensed images/alpha']:
        a_files = os.listdir(old_folder)
        for a_file in a_files:
            src_path = os.path.join(old_folder, a_file)
            dest_path = os.path.join(a_test_path, a_file)
            shutil.move(src_path, dest_path)

    if not os.path.exists(out_test_path):
        os.makedirs(out_test_path)
        do_composite_test()
