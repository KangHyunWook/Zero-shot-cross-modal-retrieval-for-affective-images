"""
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
from annoy import AnnoyIndex

import argparse 
import os
import cv2

import seaborn as sns
from sklearn.metrics import pairwise

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.

import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

"""
Postcondition: returns class semantic representation
        corresponding to given emotion embedded by BERT
Preconditifon:
    @Parameters:
    @emotion: can be either anp(adjective noun pairs) or 
        single emotion
"""

def getBERTfeatures(emotion):
    outputs=bert(preprocess([emotion]))
    sem_rep=np.array(outputs['pooled_output'][0], dtype=np.float32)
    return sem_rep

"""Set Up"""


parser=argparse.ArgumentParser()
 
parser.add_argument('--input_path', required=True, 
                              help='input path for test \
                                          emotional images')
args=parser.parse_args()
args=vars(args)


BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" # @param {type: "string"} ["https://tfhub.dev/google/experts/bert/wiki_books/2", "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2", "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2", "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",  "https://tfhub.dev/google/experts/bert/pubmed/2", "https://tfhub.dev/google/experts/bert/pubmed/squad2/2"]
# Preprocessing must match the model, but all the above use the same.
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

preprocess= hub.load(PREPROCESS_MODEL)
bert = hub.load(BERT_MODEL)

image_list=[] #contains emotional images
label_list=[] #contains corresponding emotional labels to the emotional images

"""read files names in input path"""

def getFileList(root):
    files=os.listdir(root)
    pathList=[]
    for file in files:
        path=os.path.join(root, file)
        if os.path.isdir(path):
            pathList.extend(getFileList(path))
        else:
            pathList.append(path)
    return pathList

def saveResults(fileList, candidates, target_emotion):
    cnt=1
    for i in candidates:
        print('i:',i)
        print(fileList[i])
        img=cv2.imread(fileList[i])
        plt.imshow(img)
        save_path='./testresult'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path=os.path.join(save_path,target_emotion)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(os.path.join(save_path,str(cnt)+fileList[i].split('\\')[-2]+'.jpg'))
        cnt+=1

def euclidean_distance_loss(y_true, y_pred):
    
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

fileList=getFileList(args['input_path'])
# imgList=[]
minsize=45

model=load_model('./model', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})

labels_list=[]
for file in fileList:
    img=image.load_img(file, target_size=(224,224))
    img=image.img_to_array(img)
    vgg_img=preprocess_input(img)   
    vgg_img=np.reshape(vgg_img,(-1,224,224,3))
    features=model.predict(vgg_img)  
    labels_list.append(features[0])

labels_annoy_index=AnnoyIndex(768, 'angular')
for i in range(len(labels_list)):
    labels_annoy_index.add_item(i, labels_list[i])

labels_annoy_index.build(20)
print('len: ', len(labels_list))
target_emotions=['furious eyes', 'joyful face']
for target_emotion in target_emotions:
    candidates=labels_annoy_index.get_nns_by_vector(getBERTfeatures(target_emotion), 3)
    saveResults(fileList, candidates, target_emotion)

#todo:


















