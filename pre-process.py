# -*- coding: utf-8 -*-

import zipfile
import os

if __name__ == '__main__':
    zip_file = 'Adobe_Deep_Matting_Dataset.zip'
    print('Extracting {}...'.format(zip_file))

    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall('.')
    zip_ref.close()



