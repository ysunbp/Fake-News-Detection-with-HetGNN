import pandas as pd
import numpy as np
import csv
import json
import os
import requests

pathway = '/rwproject/kdd-db/20-rayw1/rumdect/Weibo_json/'
pd.set_option('display.max_rows', 200)

f = open('/rwproject/kdd-db/20-rayw1/rumdect/Weibo.txt')
event_data = f.readlines()
event_label = []
for i in event_data:
    event = i.split('\t')
    event_label.append([event[0][4:], event[1][-1], event[2].split(' ')])
event_data = pd.DataFrame(event_label, columns=['event_id', 'rumor', 'posts'])

event_retweet = list(event_data.event_id)

file4 = open("uid_udescription.txt","a")

for filename in event_retweet:
    f = open(pathway + filename + '.json')
    data = json.load(f)
    data = pd.DataFrame(data)
    #print(data.uid)
    for i in range(len(data)):
        line1 = []
        line1.append(str(data.iloc[i].uid))
        line1.append(": ")
        line1.append(str(data.iloc[i].user_description))
        line1 = ''.join(line1)+'\n'
        print(line1)
        file4.write(str(line1))
file4.close()       