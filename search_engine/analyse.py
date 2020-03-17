import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', -1)

data = pd.read_excel("../Raw_Data/basketball.xlsx")

count = 0

print(len(data["TweetID"]))

for row in data['TweetText']:
    words = row.split(' ')
    for i in words:
        if("http" in i):
            words.remove(i)
    count = count + len(words)

print(count)

# print(data["TweetText"][2:12])