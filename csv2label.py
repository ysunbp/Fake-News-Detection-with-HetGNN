import pandas as pd

df = pd.read_csv("F:\\rumdect\\df.csv")

def find_label(id):
    a = (df[df['Id'] == int(id)]['Label'])
    if len(a)==0:
        print('wrong id')
    else:
        for i in a:
            if i == 1:
                return True
            else:
                return False
        
def batch_labels(ids):
    labels = []
    for id in ids:
        labels.append(find_label(id))
    return labels
