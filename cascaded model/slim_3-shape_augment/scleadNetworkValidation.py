'''
Created on Oct 10, 2018

@author: deeplearning
'''

import tensorflow as tf
import numpy as np
import constants as ct
from scleadNetworkArchitecture import forward_propagation
from readImageFromTFRecord import readImageFromTFRecord
from writeAndReadFiles import readInfoFromFile
from loadImageAndConvertToTFRecord import loadImageAndConvertToTFRecord, isFileExist
import time
import cv2
import os

def validate_network():
    
    if not ct.VALIDATION_PERCENTAGE:
        loadImageAndConvertToTFRecord(test_percentage=0,validation_percentage=100,inputDataDir=ct.TEST_DATASET_PATH,
                                      infoSavePath=ct.TEST_INFOMATION_PATH,tfrecordPath=ct.TEST_TFRECORD_DIR)
        dataSetSizeList = readInfoFromFile(ct.TEST_INFOMATION_PATH)
    else:
        dataSetSizeList = readInfoFromFile(ct.INFORMATION_PATH)
#     dataSetSizeList = readInfoFromFile(ct.INFORMATION_PATH)
    validation_image_num = int(dataSetSizeList['validation'])
    image_inputs=tf.placeholder(tf.float32, (1,None,None,ct.IMAGE_CHANNEL*2), 'validation_inputs')

   
#     image_inputs = tf.cond(tf.equal(input_shape_flag,0),lambda:image_inputs_256_256,
#                         lambda:tf.cond(tf.equal(input_shape_flag,1),lambda:image_inputs_181_362,lambda:image_inputs_362_181))      
#     image_inputs=tf.placeholder(tf.float32, (1,input_size[0],input_size[1],ct.IMAGE_CHANNEL*2), 'validation_inputs')
    label_inputs =tf.placeholder(tf.float32,(1,ct.CLASS_NUM), 'validation_outputs')

    nn_output,layer_outputs_tensor = forward_propagation(image_inputs,is_training=False)
    label_value_tensor = tf.argmax(label_inputs,1)
    pred_value_tensor = tf.argmax(nn_output,1)
#     correct_prediction = tf.equal(tf.argmax(nn_output,1), tf.argmax(label_inputs,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
   
    image_tensor,label_tensor,proportion_tensor= readImageFromTFRecord(ct.CATELOGS[2],tfrecord_dir= ct.TEST_TFRECORD_DIR,num_epochs=None)
    saver = tf.train.Saver()
    with tf.Session() as sess :
         
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
   
        while(True):
            TP = 0
            FN = 0
            FP = 0
            TN = 0
            ckpt = tf.train.get_checkpoint_state(ct.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                if int(global_step)>ct.STEPS+1:
                    break 
                for i in range(validation_image_num):
                    test_image, test_label, proportion= sess.run([image_tensor,label_tensor,proportion_tensor])
                    if proportion<1/1.5:
                        input_size = (360,180)
                    elif proportion >1.5:
#                         input_shape_flag = 2
                        input_size = (180,360)
                    else:
#                         input_shape_flag = 0
                        input_size = (256,256)
                        
#                     feature_map_path=os.path.join(ct.FEATURE_MAP,str(i))
#                     isFileExist(feature_map_path)
                    test_image = cv2.resize(test_image,input_size,interpolation=cv2.INTER_LINEAR)
#                     a,b = cv2.split(test_image)
#                     cv2.imwrite(os.path.join(feature_map_path,'hist.jpg'),a*255)
#                     cv2.imwrite(os.path.join(feature_map_path,'curr.jpg'),b*255)
                    
#                     cv2.namedWindow('1',0)   
#                     cv2.namedWindow('2',0)
#                     cv2.imshow('1',a)
#                     cv2.imshow('2',b) 
#                     cv2.waitKey()

                    pred,label,layer_outputs = sess.run([pred_value_tensor,label_value_tensor,layer_outputs_tensor], feed_dict= {image_inputs:[test_image],label_inputs:[test_label]})
#                     
#                     feature_map = layer_outputs['resnet_v2_50/conv1']
#                     predict = layer_outputs['prediction']
#                     with open(os.path.join(feature_map_path,'predict'),'w') as f:
#                             f.write(str(predict))
#                             f.write('\n')
# 
#                     feature_map = np.squeeze(feature_map)
#                     feature_map = cv2.split(feature_map)
# #                     cv2.namedWindow('3',0)
#                     for i in range(len(feature_map)): 
#                             cv2.imwrite(os.path.join(feature_map_path,str(i)+'.jpg'),feature_map[i]*255)
# #                         cv2.imshow('3',feature_map[i])
# #                         cv2.waitKey()
                    
                    if label[0]:
                        if pred[0]:
                            TP+=1
                        else:
                            FN+=1
                    else:
                        if not pred[0]:
                            TN+=1
                        else:
                            FP+=1
                accuracy = (TP+TN)/(TP+FN+TN+FP)
                precision = TP/(TP+FP+1e-8)
                recall = TP/(TP+FN)
                f1 = 2*precision*recall/(precision+recall+1e-8)
                print('after %s iteration, the  accuracy is %g,precision is %g,recall is %g,F1 is %g'%(global_step,accuracy,precision,recall,f1))
                
            else:
                print('no model')
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             print(sess.run(update_ops))
            print('running..........')
            time.sleep(200)
        coord.request_stop()
        coord.join(threads) 
                           
if __name__ == '__main__':
#     for i in range(480+5):
#         print('after %d min, the training will be start'%(480+5-i))
#         time.sleep(60)
    with tf.device('/cpu:0'):            
        validate_network()      
                                                              
