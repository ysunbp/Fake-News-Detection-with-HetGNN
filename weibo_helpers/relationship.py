#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
import json
import os
import requests


# In[2]:


pathway = '/rwproject/kdd-db/20-rayw1/rumdect/Weibo_json/'
pd.set_option('display.max_rows', 200)


# In[3]:


f = open('/rwproject/kdd-db/20-rayw1/rumdect/Weibo.txt')
event_data = f.readlines()
event_label = []
for i in event_data:
    event = i.split('\t')
    event_label.append([event[0][4:], event[1][-1], event[2].split(' ')])
event_data = pd.DataFrame(event_label, columns=['event_id', 'rumor', 'posts'])


# In[4]:


print(event_data.head())


# In[5]:
"""

file1 = open("user_tweet.txt","a") 
file2 = open("tweet_user.txt","a") 
"""

# In[ ]:


event_retweet = list(event_data.event_id)
"""
for filename in event_retweet:
    f = open(pathway + filename + '.json')
    data = json.load(f)
    data = pd.DataFrame(data)
    #print(data.uid)
    for i in range(len(data)):
        line1 = []
        line1.append(str(data.iloc[i].uid))
        line1.append(": ")
        line1.append(str(data.iloc[i].id))
        line1 = ''.join(line1)+'\n'
        print(line1)
        line2 = []
        line2.append(str(data.iloc[i].id))
        line2.append(": ")
        line2.append(str(data.iloc[i].uid))
        line2 = ''.join(line2)+'\n'
        print(line2)
        file1.write(str(line1))
        file2.write(str(line2))
        #print(data.head())
"""

# In[ ]:


file3 = open("user_user.txt","a")


# In[ ]:


for filename in event_retweet:
    f = open(pathway + filename + '.json')
    data = json.load(f)
    data = pd.DataFrame(data)
    #print(data.uid)
    line = []
    for i in range(len(data)):
        if i == 0:
            line.append(str(data.iloc[i].uid))
            line.append(": ")
        elif i < len(data)-1:
            line.append(str(data.iloc[i].uid))
            line.append(", ")
        else:
            line.append(str(data.iloc[i].uid))
    
    line = ''.join(line)+'\n'
    print(line)
    file3.write(str(line))

