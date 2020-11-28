#!/usr/bin/env python
# coding: utf-8

# ## Image Captioning
# - Generating Captions for Images

# ### Steps 
# - Data collection
# - Understanding the data
# - Data Cleaning
# - Loading the training set
# - Data Preprocessing — Images
# - Data Preprocessing — Captions
# - Data Preparation using Generator Function
# - Word Embeddings
# - Model Architecture
# - Inference

# In[166]:


# import all the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[167]:


# Read Text Captions

def readTextFile(path):
    with open(path) as f:
        captions = f.read()
    return captions
    


# In[23]:


captions  = readTextFile("./Data/Flickr_TextData/Flickr8k.token.txt")
captions = captions.split('\n')[:-1]


# In[24]:


print(len(captions))            # No of captions


# In[33]:


first,second  = captions[0].split('\t')  # Splitting no of captions so that each line has only one caption.
print(first.split(".")[0])
print(second)


# In[25]:


# Dictionary to Map each Image with the list of captions it has


# In[34]:


descriptions = {}

for x in captions:
    first,second = x.split('\t')
    img_name = first.split(".")[0]
    
    #if the image id is already present or not
    if descriptions.get(img_name) is None:
        descriptions[img_name] = []
    
    descriptions[img_name].append(second)


# In[35]:


descriptions["1000268201_693b08cb0e"]    # to get captions mapped with this particular image


# In[40]:


IMG_PATH = "Data/Images/"                # to see the image whether the captions generated above are relevant 
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(IMG_PATH+"1000268201_693b08cb0e.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()


# ### Data Cleaning
# 

# In[51]:


def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+"," ",sentence)
    sentence = sentence.split()
    
    sentence  = [s for s in sentence if len(s)>1]
    sentence = " ".join(sentence)
    return sentence


# In[52]:


clean_text("A cat is sitting over the house # 64")     # removing words like 'a' 


# In[53]:


# Clean all Captions
for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i])  # cleaning the ith caption


# In[54]:


descriptions["1000268201_693b08cb0e"]


# In[56]:


# Write the data to text file
with open("descriptions_1.txt","w") as f:
    f.write(str(descriptions))


# ### Vocabulary 

# In[63]:


descriptions = None
with open("descriptions_1.txt",'r') as f:
    descriptions= f.read()
    
json_acceptable_string = descriptions.replace("'","\"")
descriptions = json.loads(json_acceptable_string)


# In[64]:


print(type(descriptions))


# In[72]:


# Vocab

vocab = set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]
    
print("Vocab Size : %d"% len(vocab))


# In[75]:


# Total No of words across all the sentences
total_words = []

for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]
    
print("Total Words %d"%len(total_words))


# In[78]:


# Filter Words from the Vocab according to certain threshold frequncy


# In[80]:


import collections               # shortlisting the unique words / removing duplicates 

counter = collections.Counter(total_words)
freq_cnt = dict(counter)
print(len(freq_cnt.keys())) 


# In[86]:


# Sort this dictionary according to the freq count
sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])

# Filter
threshold = 10
sorted_freq_cnt  = [x for x in sorted_freq_cnt if x[1]>threshold]   # removing words with frequency less than 10
total_words = [x[0] for x in sorted_freq_cnt]


# In[89]:


print(len(total_words))           # final vocab 


# ### Prepare Train/Test Data

# In[93]:


train_file_data = readTextFile("Data/Flickr_TextData/Flickr_8k.trainImages.txt")
test_file_data = readTextFile("Data/Flickr_TextData/Flickr_8k.testImages.txt")


# In[99]:


train = [row.split(".")[0] for row in train_file_data.split("\n")[:-1]]
test = [row.split(".")[0] for row in test_file_data.split("\n")[:-1]]


# In[103]:


train[:5]


# In[105]:


# Prepare Description for the Training Data
# Tweak - Add <s> and <e> token to our training data
train_descriptions = {}

for img_id in train:
    train_descriptions[img_id] = []
    for cap in descriptions[img_id]:
        cap_to_append = "startseq "  + cap + " endseq"
        train_descriptions[img_id].append(cap_to_append)


# In[106]:


train_descriptions["1000268201_693b08cb0e"]


# ### Transfer Learning
# - Images --> Features
# - Text ---> Features 

# ### Step - 1 Image Feature Extraction

# In[109]:


model = ResNet50(weights="imagenet",input_shape=(224,224,3))
model.summary()


# In[112]:


model_new = Model(model.input,model.layers[-2].output)


# In[114]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img


# In[125]:


#img = preprocess_img(IMG_PATH+"1000268201_693b08cb0e.jpg")
#plt.imshow(img[0])
#plt.axis("off")
#plt.show()


# In[134]:


def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_new.predict(img)
    
    feature_vector = feature_vector.reshape((-1,))
    #print(feature_vector.shape)
    return feature_vector


# In[135]:


encode_image(IMG_PATH+"1000268201_693b08cb0e.jpg")


# In[136]:


start = time()
encoding_train = {}
#image_id -->feature_vector extracted from Resnet Image

for ix,img_id in enumerate(train):
    img_path = IMG_PATH+"/"+img_id+".jpg"
    encoding_train[img_id] = encode_image(img_path)
    
    if ix%100==0:
        print("Encoding in Progress Time step %d "%ix)
        
end_t = time()
print("Total Time Taken :",end_t-start)


# In[143]:


get_ipython().system('mkdir saved')


# In[145]:


# Store everything to the disk 
with open("saved/encoded_train_features.pkl","wb") as f:
    pickle.dump(encoding_train,f)


# In[138]:


start = time()
encoding_test = {}
#image_id -->feature_vector extracted from Resnet Image

for ix,img_id in enumerate(test):
    img_path = IMG_PATH+"/"+img_id+".jpg"
    encoding_test[img_id] = encode_image(img_path)
    
    if ix%100==0:
        print("Test Encoding in Progress Time step %d "%ix)
        
end_t = time()
print("Total Time Taken(test) :",end_t-start)


# In[146]:


with open("saved/encoded_test_features.pkl","wb") as f:
    pickle.dump(encoding_test,f)


# ### Data pre-processing for Captions

# In[149]:


# Vocab
len(total_words)


# In[155]:


word_to_idx = {}
idx_to_word = {}

for i,word in enumerate(total_words):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word


# In[159]:


#word_to_idx["dog"]
#idx_to_word[1]
print(len(idx_to_word))


# In[162]:


# Two special words
idx_to_word[1846] = 'startseq'
word_to_idx['startseq'] = 1846

idx_to_word[1847] = 'endseq'
word_to_idx['endseq'] = 1847

vocab_size = len(word_to_idx) + 1
print("Vocab Size",vocab_size)


# In[164]:


max_len = 0 
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        max_len = max(max_len,len(cap.split()))
        
print(max_len)


# ### Data Loader (Generator)

# In[165]:


def data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size):
    X1,X2, y = [],[],[]
    
    n =0
    while True:
        for key,desc_list in train_descriptions.items():
            n += 1
            
            photo = encoding_train[key+".jpg"]
            for desc in desc_list:
                
                seq = [word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1,len(seq)):
                    xi = seq[0:i]
                    yi = seq[i]
                    
                    #0 denote padding word
                    xi = pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi = to_categorcial([yi],num_classes=vocab_size)[0]
                    
                    X1.append(photo)
                    X2.append(xi)
                    y.append(yi)
                    
                if n==batch_size:
                    yield [[np.array(X1),np.array(X2)],np.array(y)]
                    X1,X2,y = [],[],[]
                    n = 0


# ## Word Embeddings 

# In[177]:


f = open("./saved/glove.6B.50d.txt",encoding='utf8')


# In[178]:


embedding_index = {}

for line in f:
    values = line.split()
    
    word = values[0]
    word_embedding = np.array(values[1:],dtype='float')
    embedding_index[word] = word_embedding
    


# In[180]:


f.close()


# In[179]:


embedding_index['apple']


# In[187]:


def get_embedding_matrix():
    emb_dim = 50
    matrix = np.zeros((vocab_size,emb_dim))
    for word,idx in word_to_idx.items():
        embedding_vector = embedding_index.get(word)
        
        if embedding_vector is not None:
            matrix[idx] = embedding_vector
            
    return matrix
        
    


# In[188]:


embedding_matrix = get_embedding_matrix()
embedding_matrix.shape


# In[190]:


#embedding_matrix[1847]


# #### Model Architecture

# In[191]:


input_img_features = Input(shape=(2048,))
inp_img1 = Dropout(0.3)(input_img_features)
inp_img2 = Dense(256,activation='relu')(inp_img1)


# In[192]:


# Captions as Input
input_captions = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)


# In[193]:


decoder1 = add([inp_img2,inp_cap3])
decoder2 = Dense(256,activation='relu')(decoder1)
outputs = Dense(vocab_size,activation='softmax')(decoder2)

# Combined Model
model = Model(inputs=[input_img_features,input_captions],outputs=outputs)


# In[194]:


model.summary()


# In[195]:


# Important Thing - Embedding Layer
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False


# In[196]:


model.compile(loss='categorical_crossentropy',optimizer="adam")


# ### Training of Model

# In[197]:


epochs = 20
batch_size = 3
steps = len(train_descriptions)//number_pics_per_batch


# In[198]:


def train():
    
    for i in range(epochs):
        generator = data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size)
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        model.save('./model_weights/model_'+str(i)+'.h5')


# In[199]:


model = load_model('./model_weights/model_9.h5')


# ## Predictions

# In[222]:


def predict_caption(photo):
    
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[230]:


# Pick Some Random Images and See Results
plt.style.use("seaborn")
for i in range(15):
    idx = np.random.randint(0,1000)
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
    photo_2048 = encoding_test[img_name].reshape((1,2048))
    
    i = plt.imread("Data/Images/"+img_name+".jpg")    # give image path 
    
    caption = predict_caption(photo_2048)
    #print(caption)
    
    plt.title(caption)
    plt.imshow(i)
    plt.axis("off")
    plt.show()
    

