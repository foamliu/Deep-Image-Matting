# -*- coding: utf-8 -*-

import zipfile
import os
import shutil
from Combined_Dataset.Training_set.Composition_code_revised import do_composite

if __name__ == '__main__':
    if not os.path.exists('Combined_Dataset'):
        zip_file = 'Adobe_Deep_Matting_Dataset.zip'
        print('Extracting {}...'.format(zip_file))

        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall('.')
        zip_ref.close()

    if not os.path.exists('train2014'):
        zip_file = 'train2014.zip'
        print('Extracting {}...'.format(zip_file))

        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall('.')
        zip_ref.close()

    training_bg_names = []
    with open('Combined_Dataset/Training_set/training_bg_names.txt') as f:
        training_bg_names = f.read().splitlines()

    # path to provided foreground images
    fg_path = 'fg/'
    # path to provided alpha mattes
    a_path = 'mask/'
    # Path to background images (MSCOCO)
    bg_path = 'bg/'
    # Path to folder where you want the composited images to go
    out_path = 'merged/'

    train_folder = 'Combined_Dataset/Training_set/'
    if not os.path.exists(bg_path):
        os.makedirs(bg_path)
        for bg_name in training_bg_names:
            src_path = os.path.join('train2014', bg_name)
            dest_path = os.path.join(bg_path, bg_name)
            shutil.move(src_path, dest_path)

    if not os.path.exists(fg_path):
        os.makedirs(fg_path)

        for old_folder in [train_folder + 'Adobe-licensed images/fg', train_folder + 'Other/fg']:
            old_folder = os.path.join()
            fg_files = os.listdir(old_folder)
            for fg_file in fg_files:
                src_path = os.path.join(old_folder, fg_file)
                dest_path = os.path.join(fg_path, fg_file)
                shutil.move(src_path, dest_path)

    if not os.path.exists(a_path):
        os.makedirs(a_path)

        for old_folder in [train_folder + 'Adobe-licensed images/alpha', train_folder + 'Other/alpha']:
            old_folder = os.path.join(train_folder, 'Adobe-licensed images/alpha')
            a_files = os.listdir(old_folder)
            for a_file in a_files:
                src_path = os.path.join(old_folder, a_file)
                dest_path = os.path.join(a_path, a_file)
                shutil.move(src_path, dest_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    do_composite()
