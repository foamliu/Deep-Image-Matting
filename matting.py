import tensorflow as tf
import numpy as np
import random
from scipy import misc,ndimage
import copy
import itertools
import os
from sys import getrefcount
import gc

trimap_kernel = [val for val in range(20,40)]
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

hard_samples = [
1,4,8,11,13,15,16,19,28,42,43,44,46,65,68,69,70,81,91,92,94,101,104,
118,137,145,152,155,156,176,187,189,191,193,198,203,208,212,215,
216,221,233,239,243,254,264,265,267,272,278,279,281,288,290,291,292,
293,298,300,301,302,309,316,320,325,337,345,346,347,369,370,374,381,
386,402,416,432,443,450,451,454,456,457,459,464,487,490,499,502,513,
514,552,555,558,559,577,580,587,593,602,608,609,613,632,634,639,640,
649,663,688,710,717,718,723,729,736,740,741,745,757,769,775,778,785,
788,805,808,815,820,834,839,840,845,846,848,860,861,864,868,870,872,
877,885,889,892,894,895
]

def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    
	with tf.variable_scope(scope):
		input_shape = pool.get_shape().as_list()
		output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

		flat_input_size = np.prod(input_shape)
		flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

		pool_ = tf.reshape(pool, [flat_input_size])
		batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
		b = tf.ones_like(ind) * batch_range
		b = tf.reshape(b, [flat_input_size, 1])
		ind_ = tf.reshape(ind, [flat_input_size, 1])
		ind_ = tf.concat([b, ind_], 1)

		ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
		ret = tf.reshape(ret, output_shape)
		return ret


def UR_center(trimap):

    target = np.where(trimap==128)
    index = random.choice([i for i in range(len(target[0]))])
    return  np.array(target)[:,index][:2]

def load_path(alpha,eps,BG,hard_mode = False):
    folders = os.listdir(alpha)
    common_paths = []
    if hard_mode:
        for folder in folders:
            if int(folder) in hard_samples: 
                images = os.listdir(os.path.join(alpha,folder))
                common_paths.extend([os.path.join(folder,image) for image in images])
    else:
        for folder in folders:
            #if int(folder)==137:
            images = os.listdir(os.path.join(alpha,folder))
            common_paths.extend([os.path.join(folder,image) for image in images])
    print(common_paths)
    alphas_abspath = [os.path.join(alpha,common_path) for common_path in common_paths]
    epses_abspath = [os.path.join(eps,common_path) for common_path in common_paths]
    BGs_abspath = [os.path.join(BG,common_path)[:-3] + 'jpg' for common_path in common_paths]
    return np.array(alphas_abspath),np.array(epses_abspath),np.array(BGs_abspath)

def load_data(batch_alpha_paths,batch_eps_paths,batch_BG_paths):
	
	batch_size = batch_alpha_paths.shape[0]
	train_batch = []
	images_without_mean_reduction = []
	for i in range(batch_size):
			
		alpha = misc.imread(batch_alpha_paths[i],'L').astype(np.float32)

		eps = misc.imread(batch_eps_paths[i]).astype(np.float32)

		BG = misc.imread(batch_BG_paths[i]).astype(np.float32)
		
		batch_i,raw_RGB = preprocessing_single(alpha, BG, eps,batch_alpha_paths[i])	
		train_batch.append(batch_i)
		images_without_mean_reduction.append(raw_RGB)
	train_batch = np.stack(train_batch)
	return train_batch[:,:,:,:3],np.expand_dims(train_batch[:,:,:,3],3),np.expand_dims(train_batch[:,:,:,4],3),train_batch[:,:,:,5:8],train_batch[:,:,:,8:],images_without_mean_reduction

def generate_trimap(trimap,alpha):

	k_size = random.choice(trimap_kernel)
	trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - ndimage.grey_erosion(alpha[:,:,0],size=(k_size,k_size)))!=0)] = 128
	#trimap[np.where((ndimage.grey_dilation(alpha[:,:,0],size=(k_size,k_size)) - alpha[:,:,0]!=0))] = 128
	return trimap

def preprocessing_single(alpha, BG, eps,name,image_size=320):

    alpha = np.expand_dims(alpha,2)
    trimap = np.copy(alpha)
    trimap = generate_trimap(trimap,alpha)

    train_data = np.zeros([image_size,image_size,8])
    crop_size = random.choice([320,480,620])
#    crop_size = 320   
    flip = random.choice([0,1])
    i_UR_center = UR_center(trimap)
    #i_UR_center = [int(alpha.shape[0]/2),int(alpha.shape[1]/2)]
    train_pre = np.concatenate([trimap,alpha,BG,eps],2)

    if crop_size == 320:
        h_start_index = i_UR_center[0] - 159
        w_start_index = i_UR_center[1] - 159
        tmp = train_pre[h_start_index:h_start_index+320, w_start_index:w_start_index+320, :]
        if flip:
            tmp = tmp[:,::-1,:]
        tmp[:,:,1] = tmp[:,:,1] / 255.0
        tmp[:,:,5:] = np.expand_dims(tmp[:,:,1],2)  * tmp[:,:,5:]  # here replace eps with FG
        raw_RGB = np.expand_dims(tmp[:,:,1],2)  * tmp[:,:,5:] + np.expand_dims((1. - tmp[:,:,1]),2) * tmp[:,:,2:5]
        reduced_RGB = raw_RGB - g_mean
        tmp = np.concatenate([reduced_RGB,tmp],2)
        train_data = tmp

    if crop_size == 480:
        h_start_index = i_UR_center[0] - 239
        w_start_index = i_UR_center[1] - 239
        tmp = train_pre[h_start_index:h_start_index+480, w_start_index:w_start_index+480, :]
        if flip:
            tmp = tmp[:,::-1,:]
        tmp1 = np.zeros([image_size,image_size,8]).astype(np.float32)
        tmp1[:,:,0] = misc.imresize(tmp[:,:,0].astype(np.uint8),[image_size,image_size],interp = 'nearest',mode='L').astype(np.float32)
        tmp1[:,:,1] = misc.imresize(tmp[:,:,1].astype(np.uint8),[image_size,image_size]).astype(np.float32) / 255.0
        tmp1[:,:,2:5] = misc.imresize(tmp[:,:,2:5].astype(np.uint8),[image_size,image_size,3]).astype(np.float32)
        tmp1[:,:,5:] = misc.imresize(tmp[:,:,5:].astype(np.uint8),[image_size,image_size,3]).astype(np.float32)
        tmp1[:,:,5:] = np.expand_dims(tmp1[:,:,1],2)  * tmp1[:,:,5:]  # here replace eps with FG        
        raw_RGB = np.expand_dims(tmp1[:,:,1],2)  * tmp1[:,:,5:] + np.expand_dims((1. - tmp1[:,:,1]),2) * tmp1[:,:,2:5]
        reduced_RGB = raw_RGB - g_mean      
        tmp1 = np.concatenate([reduced_RGB,tmp1],2)
        train_data = tmp1

    if crop_size == 620:
        h_start_index = i_UR_center[0] - 309
        #boundary security
        if h_start_index<0:
            h_start_index = 0
        w_start_index = i_UR_center[1] - 309
        if w_start_index<0:
            w_start_index = 0
        tmp = train_pre[h_start_index:h_start_index+620, w_start_index:w_start_index+620, :]
        if flip:
            tmp = tmp[:,::-1,:]
        tmp1 = np.zeros([image_size,image_size,8]).astype(np.float32)
        tmp1[:,:,0] = misc.imresize(tmp[:,:,0].astype(np.uint8),[image_size,image_size],interp = 'nearest',mode='L').astype(np.float32)
        tmp1[:,:,1] = misc.imresize(tmp[:,:,1].astype(np.uint8),[image_size,image_size]).astype(np.float32) / 255.0
        tmp1[:,:,2:5] = misc.imresize(tmp[:,:,2:5].astype(np.uint8),[image_size,image_size,3]).astype(np.float32)
        tmp1[:,:,5:] = misc.imresize(tmp[:,:,5:].astype(np.uint8),[image_size,image_size,3]).astype(np.float32)
        tmp1[:,:,5:] = np.expand_dims(tmp1[:,:,1],2)  * tmp1[:,:,5:]  # here replace eps with FG        
        raw_RGB = np.expand_dims(tmp1[:,:,1],2)  * tmp1[:,:,5:] + np.expand_dims((1. - tmp1[:,:,1]),2) * tmp1[:,:,2:5]
        reduced_RGB = raw_RGB - g_mean      
        tmp1 = np.concatenate([reduced_RGB,tmp1],2)
        train_data = tmp1
    train_data = train_data.astype(np.float32)
#    misc.imsave('./train_alpha.png',train_data[:,:,4])
    return train_data,raw_RGB

def load_alphamatting_data(test_alpha):
	rgb_path = os.path.join(test_alpha,'rgb')
	trimap_path = os.path.join(test_alpha,'trimap')
	alpha_path = os.path.join(test_alpha,'alpha')	
	images = os.listdir(trimap_path)
	test_num = len(images)
	all_shape = []
	rgb_batch = []
	tri_batch = []
	alp_batch = []
	for i in range(test_num):
		rgb = misc.imread(os.path.join(rgb_path,images[i]))
		trimap = misc.imread(os.path.join(trimap_path,images[i]),'L')
		alpha = misc.imread(os.path.join(alpha_path,images[i]),'L')/255.0
		all_shape.append(trimap.shape)
		rgb_batch.append(misc.imresize(rgb,[320,320,3])-g_mean)
		trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)
		tri_batch.append(np.expand_dims(trimap,2))
		alp_batch.append(alpha)
	return np.array(rgb_batch),np.array(tri_batch),np.array(alp_batch),all_shape,images

def load_validation_data(vali_root):
	alpha_dir = os.path.join(vali_root,'alpha')
	RGB_dir = os.path.join(vali_root,'RGB')
	images = os.listdir(alpha_dir)
	test_num = len(images)
	
	all_shape = []
	rgb_batch = []
	tri_batch = []
	alp_batch = []

	for i in range(test_num):
		rgb = misc.imread(os.path.join(RGB_dir,images[i]))
		alpha = misc.imread(os.path.join(alpha_dir,images[i]),'L') 
		trimap = generate_trimap(np.expand_dims(np.copy(alpha),2),np.expand_dims(alpha,2))[:,:,0]
		alpha = alpha / 255.0
		all_shape.append(trimap.shape)
		rgb_batch.append(misc.imresize(rgb,[320,320,3])-g_mean)
		trimap = misc.imresize(trimap,[320,320],interp = 'nearest').astype(np.float32)
		tri_batch.append(np.expand_dims(trimap,2))
		alp_batch.append(alpha)
	return np.array(rgb_batch),np.array(tri_batch),np.array(alp_batch),all_shape,images
