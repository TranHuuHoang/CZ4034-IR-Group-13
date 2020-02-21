import pandas as pd

data = pd.read_csv("data/test.csv", sep=";")

print("Shape of data: " + str(data.shape))

print(data.columns)

