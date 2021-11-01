"""
Date: 2021-Oct-2
Description: This program trains the emotional images that have their
            labels(ANP) are embedded class semantic features by BERT.
            The trained model is saved in the given path
            usage: [prog] [train_data_path] [output_model_path]
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.losses import cosine_similarity
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras import Model

import argparse 
import os

import seaborn as sns
from sklearn.metrics import pairwise

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Imports TF ops for preprocessing.

import numpy as np
import matplotlib.pyplot as plt
import cv2

import keras.backend as K

def euclidean_distance_loss(y_true,y_pred):
    return K.sqrt(K.sum(K.square(y_pred-y_true), axis=-1))

"""
Postcondition: returns class semantic representation
        corresponding to given emotion embedded by BERT
Precondition:
    @Parameters:
    @emotion: can be either anp(adjective noun pairs) or 
        single emotion
"""

def getBERTfeatures(emotion):
    outputs=bert(preprocess([emotion]))
    sem_rep=np.array(outputs['pooled_output'][0], dtype=np.float32)
    return sem_rep

"""Set Up"""
parser = argparse.ArgumentParser()

parser.add_argument('--train', required=True)

parser.add_argument('--output', required=True)

args=parser.parse_args()    

args=vars(args)

input_path=args['train']

saved_model_path=args['output']

BERT_MODEL = "https://tfhub.dev/google/experts/bert/wiki_books/2" # @param {type: "string"} ["https://tfhub.dev/google/experts/bert/wiki_books/2", "https://tfhub.dev/google/experts/bert/wiki_books/mnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qnli/2", "https://tfhub.dev/google/experts/bert/wiki_books/qqp/2", "https://tfhub.dev/google/experts/bert/wiki_books/squad2/2", "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",  "https://tfhub.dev/google/experts/bert/pubmed/2", "https://tfhub.dev/google/experts/bert/pubmed/squad2/2"]
# Preprocessing must match the model, but all the above use the same.
PREPROCESS_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"

preprocess= hub.load(PREPROCESS_MODEL)
bert = hub.load(BERT_MODEL)


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

def getTrainData(fileList):
    images_vgg16=[]
    labels_list=[]
    for file in fileList:
        splits=file.split(os.path.sep)[-2].split('_')
        img_pil = image.load_img(file, target_size=(224,224))
        img_raw = image.img_to_array(img_pil)
        img=preprocess_input(img_raw)
        images_vgg16.append(img)
        ANP=splits[0]+' '+splits[1]
        labels_list.append(getBERTfeatures(ANP))
    return np.array(images_vgg16), np.array(labels_list)

def buildModel(input_shape, intermediate_dim=2000, word_embedding_dim=768):
    vgg16=VGG16(input_shape=input_shape)
    x=vgg16.get_layer('fc2').output
    for layer in vgg16.layers:
        layer.trainable=False
    x=Dense(intermediate_dim, name="dense1")(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(word_embedding_dim, name="dense2")(x)
    outputs=BatchNormalization()(x)
    model=Model(inputs=[vgg16.input], outputs=outputs)  
    return model    

"""read files names in input path"""

fileList=getFileList(input_path)

# imgList=[]
minsize=45

images_vgg16, labels_list = getTrainData(fileList)
#todo:

epochs=30
batch_size=10
X, y=shuffle(images_vgg16, labels_list, random_state=7)

print('images len:',len(images_vgg16))

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=7)

checkpointer=ModelCheckpoint(filepath='best.hdf5', monitor='val_accuracy', save_best_only=True)

model=buildModel(images_vgg16.shape[1:])
sgd=optimizers.SGD(learning_rate=0.001, decay=1e-6, nesterov=True)
model.compile(loss=euclidean_distance_loss, optimizer=sgd)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
callbacks=[checkpointer])

model.save('./model')

print("files len:", len(fileList))

































