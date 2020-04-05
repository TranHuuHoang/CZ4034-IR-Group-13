import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', -1)

data = pd.read_excel("./Classification - Naive Bayes and SVM/healthcareoutput.xlsx")

count = 0

print(data.columns)

words = []
for row in data['TweetText']:
    try:
        word = row.split(' ')
        for i in word:
            if("http" in i):
                word.remove(i)
        words.extend(word)
    except:
        pass

count = len(words)
count_unique = len(set(words))

print(count)
print(count_unique)

# print(data["TweetText"][2:12])