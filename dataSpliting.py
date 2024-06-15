import os
import pandas as pd


directory = 'News Articles'


file_contents = []
categories = []


for subdir, _, files in os.walk(directory):
    for file in files:
        file_path = os.path.join(subdir, file)
        with open(file_path, 'r') as f:
            content = f.read()
            file_contents.append(content)
            category = os.path.basename(subdir)
            categories.append(category)


df = pd.DataFrame({"categories" : categories , "articles" : file_contents})
df.to_csv("CategoryData.csv")
print(df)