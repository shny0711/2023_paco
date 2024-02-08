import pandas as pd


data={}
for i in range(10):
    data[f"{i}番目"] = [i/10, i*2]

    #間違い


df = pd.DataFrame(data)
df.to_csv("testdata/test.csv")
