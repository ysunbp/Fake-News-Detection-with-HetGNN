import pandas as pd

f = open('F:\\rumdect\\Weibo.txt')
#doc_list = open('F:\\rumdect\\doc_list','a')
lines = f.readlines()
data = []
i = 0
for line in lines:
    elements = line.split() #['event','label','p1','p2'..]
    #print(len(elements))
    for j in range(len(elements)):
        if j == 0:
            continue
        elif j == 1:
            label = elements[j][-1]
        else:
            data.append([elements[j], label])
    i+=1
    print(i)
df = pd.DataFrame(data, columns = ['Id', 'Label'])
df.to_csv("F:\\rumdect\\df.csv", index = False)
