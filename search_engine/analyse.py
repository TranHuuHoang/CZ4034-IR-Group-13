import pandas as pd

data = pd.read_excel("Raw_Data/basketball.xlsx")

count = 0
for row in data['TweetText']:
    words = row.split(' ')
    for i in words:
        if("http" in i):
            words.remove(i)
    count = count + len(words)

print(count)

print(type(data['Date(SGT - 9)'][2]))
