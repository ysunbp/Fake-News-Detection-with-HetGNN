import json
import os
import urllib.request
import http.cookiejar

data_dir = "F:\\rumdect\\Weibo\\"
all_json = os.listdir(data_dir)
i = 0
k = 0
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [('User-Agent' , 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 ')]
urllib.request.install_opener(opener)
for j_name in all_json:
    if  i>=2579 :
        j = open(data_dir+j_name,'rb')
        info = json.load(j)
        print(i)
        if (info[0]['picture'] != None):
            urllib.request.urlretrieve(info[0]['picture'], "F:\\img\\%s.jpg" % (info[0]['id']))
            k += 1

    i += 1

print("Total number of weibo:")
print(i)
print("Total number of images:")
print(k)