'''
Created on Oct 8, 2018

@author: deeplearning
'''
import os
import tensorflow as tf
import numpy as np
import constants as ct
from readImageFromTFRecord import readImageBatchFromTFRecord
from loadImageAndConvertToTFRecord import loadImageAndConvertToTFRecord, isFileExist
from scleadNetworkArchitecture import forward_propagation
import time
import cv2
import tensorflow.contrib.slim as slim

def resize_batch(batch,size):
    batch_resized=[]
    for i in range(ct.BATCH_SIZE):
        batch_resized.append(cv2.resize(batch[i],size,interpolation=cv2.INTER_LINEAR))
    return np.array(batch_resized)

def train_network(training_image_num):
    global_step = tf.Variable(0, trainable=False)
    image_inputs=tf.placeholder(tf.float32, (ct.BATCH_SIZE,None,None,ct.IMAGE_CHANNEL*2), 'inputs')     
    label_inputs =tf.placeholder(tf.float32,(ct.BATCH_SIZE,ct.CLASS_NUM), 'outputs')
    
#     selected_input_entrance = global_step%3
#     image_inputs = tf.cond(tf.equal(selected_input_entrance,0),lambda:image_inputs_256_256,
#                         lambda:tf.cond(tf.equal(selected_input_entrance,1),lambda:image_inputs_181_362,lambda:image_inputs_362_181))
    nn_output,_ = forward_propagation(image_inputs)
#     output_max = tf.reduce_max(nn_output, axis=1)
#     nn_output = tf.clip_by_value(nn_output,1e-8,1.0)
#     nn_output = nn_output/50000
#     temp =tf.nn.softmax(nn_output)
#     nn_softmax = tf.clip_by_value(tf.nn.softmax(nn_output),1e-10,1.0)
#     cross_entropy_loss = label_inputs * tf.log(nn_softmax)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=nn_output, labels=tf.argmax(label_inputs,1))
    cross_entropy_loss_mean = tf.reduce_mean(cross_entropy_loss)

#     loss_func = cross_entropy_loss_mean
    learning_rate = tf.train.exponential_decay(ct.LEARNING_RATE_INIT, global_step, training_image_num/ct.BATCH_SIZE, ct.LEARNING_DECAY_RATE)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_step = slim.learning.create_train_op(cross_entropy_loss_mean,optimizer,global_step=global_step)
#     train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_func, global_step)
    
   

    image_batch_tensor,label_batch_tensor= readImageBatchFromTFRecord(ct.CATELOGS[0])
    saver = tf.train.Saver()
    
    isFileExist(ct.MODEL_SAVE_PATH)
    with tf.Session() as sess:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()
      
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        start_time = time.time()
        for i in range(ct.STEPS+1):
#             ct.INPUT_SIZE_CURR = ct.INPUT_SIZE[i%3]
            image_batch, label_batch = sess.run([image_batch_tensor,label_batch_tensor])
            
#             a,b = cv2.split(image_batch[93])
#             cv2.namedWindow('1',0)   
#             cv2.namedWindow('2',0)
#             cv2.imshow('1',a)
#             cv2.imshow('2',b) 
#             cv2.waitKey()
            
            if i%3==0:
                img_size =(256,256)
            elif i%3==1:
                img_size =(360,180)
#                 img_size =(256,256)
            else:
                img_size =(180,360)
#                 img_size =(256,256)   

            image_batch = resize_batch(image_batch,img_size)
            

#             a,b = cv2.split(image_batch[93])
#             cv2.namedWindow('3',0)   
#             cv2.namedWindow('4',0)
#             cv2.imshow('3',a)
#             cv2.imshow('4',b) 
#             cv2.waitKey()
            
            _,loss_val,step = sess.run([train_step, cross_entropy_loss_mean,global_step], 
                                       feed_dict= {image_inputs:image_batch,label_inputs:label_batch})
            if not (i%100):
                print('after %d iteration, loss is %g'%(step,loss_val))
                if not (i%1000):
                    saver.save(sess,os.path.join(ct.MODEL_SAVE_PATH,ct.MODEL_NAME) ,global_step)
                    duration = time.time()-start_time
                    print('duration is %g'%duration)
           


#             print(label_batch)
#                 output = sess.run(nn_softmax,
#                 feed_dict= {image_inputs:image_batch,label_inputs:label_batch})    
#                 print(output)
              
#             output = sess.run(temp, 
#             feed_dict= {image_inputs:image_batch,label_inputs:label_batch})    
#             print(output)  
        coord.request_stop()
        coord.join(threads)

def main(_):
#     for i in range(480):
#         print('after %d min, the training will be start'%(480-i))
#         time.sleep(60)
    training_image_num=loadImageAndConvertToTFRecord()
#     training_image_num=3922
    train_network(training_image_num)

if __name__ == '__main__' :
    tf.app.run()