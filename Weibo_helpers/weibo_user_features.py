#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import numpy as np
import json

# In[ ]:


pathway = '../rumdect/weibo_json/'
#pd.set_option('display.max_rows', 200)


# In[ ]:


event_data = pd.read_csv('../rumdect/weibo_events_label.csv')
event_retweet = list(event_data.event_id)


# In[ ]:


for filename in event_retweet:
    f = open(pathway + str(filename) + '.json')

    data = json.load(f)
    data = pd.DataFrame(data)

    # create an array to store the info of retweet users
    uid_list = list()
    retweet_users = list()
    for i in range(len(data)):
        repost_count = str(data.reposts_count[i])
        bi_followers_count = str(data.bi_followers_count[i])
        num_friends = str(data.friends_count[i])
        num_word_dscptn = str(len(data.user_description[i]))
        num_word_name = str(len(data.screen_name[i]))
        num_followers = str(data.followers_count[i])
        # encoding screen name and self-description
        num_statuses = str(data.statuses_count[i])
        verified = str(data.verified[i].astype(int))
        geo_position = str(data.user_geo_enabled[i].astype(int))
        # time difference between original post and the repost
        # length of the retweet path (i didnt add it because all path length is 1 in this case)
        time = str(data.t[i] - data.user_created_at[i])
        num_favorite = str(data.favourites_count[i])
        num_comment = str(data.comments_count[i])
        
        retweet_users.append([repost_count, bi_followers_count, num_friends,
                              num_word_dscptn, num_word_name, num_followers, num_statuses,
                              verified, geo_position, time, num_favorite, num_comment])
        uid_list.append(data.uid[i])
    
    with open('../rumdect/weibo_user_feature/' + str(filename) + ".txt", "w") as ofile:
        for i in range(len(retweet_users)):
            ofile.write(str(uid_list[i]))
            ofile.write(' ')
            ofile.write(' '.join(retweet_users[i]))
            ofile.write('\n')

    ofile.close()
    

