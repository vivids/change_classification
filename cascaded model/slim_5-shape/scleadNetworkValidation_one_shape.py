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
                    
                    result=[0,0,0,0,0]
                    prediction = 0
                    input_size = ((360,180),(180,360),(256,256),(442,148),(148,442))
                    for j in range(5):
                        test_image = cv2.resize(test_image,input_size[j],interpolation=cv2.INTER_LINEAR)
                        pred,label,layer_outputs = sess.run([pred_value_tensor,label_value_tensor,layer_outputs_tensor], feed_dict= {image_inputs:[test_image],label_inputs:[test_label]})
                        result[j]=pred[0];
                    if sum(result)>=3:
                        prediction = 1    
                    
                    if label[0]:
                        if prediction:
                            TP+=1
                        else:
                            FN+=1
                    else:
                        if not prediction:
                            TN+=1
                        else:
                            FP+=1
                accuracy = (TP+TN)/(TP+FN+TN+FP)
                precision = TP/(TP+FP+1e-8)
                recall = TP/(TP+FN)
                f1 = 2*precision*recall/(precision+recall+1e-8)
                print('after %s iteration, the  accuracy is %g,precision is %g,recall is %g,F1 is %g'%(global_step,accuracy,precision,recall,f1))
                
                TP = 0
                FN = 0
                FP = 0
                TN = 0
                for i in range(validation_image_num):
                    test_image, test_label, proportion= sess.run([image_tensor,label_tensor,proportion_tensor])
                    result=[0,0,0]
                    prediction = 0
                    input_size = ((360,180),(180,360),(256,256))
                    for j in range(3):
                        test_image = cv2.resize(test_image,input_size[j],interpolation=cv2.INTER_LINEAR)
                        pred,label,layer_outputs = sess.run([pred_value_tensor,label_value_tensor,layer_outputs_tensor], feed_dict= {image_inputs:[test_image],label_inputs:[test_label]})
                        result[j]=pred[0];
                    if sum(result)>=2:
                        prediction = 1    
                    
                    if label[0]:
                        if prediction:
                            TP+=1
                        else:
                            FN+=1
                    else:
                        if not prediction:
                            TN+=1
                        else:
                            FP+=1
                accuracy = (TP+TN)/(TP+FN+TN+FP)
                precision = TP/(TP+FP+1e-8)
                recall = TP/(TP+FN)
                f1 = 2*precision*recall/(precision+recall+1e-8)
                print('2--after %s iteration, the  accuracy is %g,precision is %g,recall is %g,F1 is %g'%(global_step,accuracy,precision,recall,f1))
                
            else:
                print('no model')
#             update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#             print(sess.run(update_ops))
            print('running..........')
            time.sleep(100)
        coord.request_stop()
        coord.join(threads) 
                           
if __name__ == '__main__':
#     for i in range(480+5):
#         print('after %d min, the training will be start'%(480+5-i))
#         time.sleep(60)
    with tf.device('/cpu:0'):            
        validate_network()      
                                                              
