#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import csv
import json
import os
import requests
import sys

# In[33]:


#pathway = '/Users/jessica/Desktop/FYP/Dataset/rumdect/Weibo_json/'
#pathway = 'Dataset/rumdect/Weibo_json/'
pathway = '../rumdect/weibo_json/'
os.chdir(pathway)
pd.set_option('display.max_rows', 200)


# In[30]:


'''
#f = open('Dataset/rumdect/Weibo.txt')
f = open('../rumdect/Weibo.txt')
event_data = f.readlines()
event_label = []
for i in event_data:
    event = i.split('\t')
    event_label.append([event[0][4:], event[1][-1], event[2].split(' ')])
event_data = pd.DataFrame(event_label, columns=['event_id', 'rumor', 'posts'])
#event_data.to_csv('weibo_events.csv')
'''


# In[31]:


event_data = pd.read_csv("../weibo_events_label.csv")
event_retweet = list(event_data.event_id)


# In[35]:


for i in range(2178, len(event_retweet)):
    filename = event_retweet[i]
    f = open(str(filename) + '.json')
    print(i)
    data = json.load(f)
    data = pd.DataFrame(data)

    for i in range(len(data)):
        try:
            url= data.user_avatar[i]
            ind = data.uid[i]
            response = requests.get(url)
            file = open('../weibo_user_protrait/' + str(ind) + '.jpg', 'wb')
            file.write(response.content)
            file.close()
        except:
            pass


# In[36]:





# In[ ]:




