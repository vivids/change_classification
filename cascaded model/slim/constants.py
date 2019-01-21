'''
Created on Oct 8, 2018

@author: deeplearning
'''
INPUT_DATA_DIR ='/home/deeplearning/datasets/alarmClassification/experiment/train'
OUTPUT_TFRECORD_DIR = 'output/tfrecord'
MODEL_SAVE_PATH = 'output/model'
MODEL_NAME = 'sclead_network_model.ckpt'
INFORMATION_PATH='output/info'
FEATURE_MAP = 'output/featuremap'
CATELOGS = ('training','testing','validation')
# CATELOGS_LABELS={'stain':0,'luminance':1,'rotation':2,'abnormal':3,'foreignBody':4,'character':5}
CATELOGS_LABELS={'0':0,'1':1}
TEST_PERCENTAGE = 0
VALIDATION_PERCENTAGE = 0
INPUT_SIZE =256

IMAGE_CHANNEL =1
NUM_THREAD=4
MIN_AFTER_DEQUEUE = 8000
BATCH_SIZE = 96
CLASS_NUM =2
LEARNING_RATE_INIT = 0.01
LEARNING_DECAY_RATE = 0.99
STEPS=80000

if not TEST_PERCENTAGE:
#     TEST_DATASET_PATH ='/home/deeplearning/datasets/alarmClassification/experiment/test'
    TEST_DATASET_PATH ='/home/deeplearning/datasets/alarmClassification/experiment/test_view'
    TEST_INFOMATION_PATH = 'output/testInfo'
    TEST_TFRECORD_DIR = 'output/tfrecord_test'
    

# slim
BLOCK1=[(150,50,1)]*2+[(150,50,2)]
BLOCK2=[(300,100,1)]*2+[(300,100,2)]
BLOCK3=[(600,200,1)]*2+[(600,200,2)]
BLOCK4=[(1200,400,1)]*1



