import pandas as pd
from eventregistry import *
import requests
from datetime import date,timedelta


today = date.today()
day_before = today - timedelta(days=2)

api_key = "53bc9038-244c-443f-b6e5-b94bebb59104"

er = EventRegistry(apiKey=api_key)

def gettingNews():
    query = QueryArticlesIter(
        dataType=["news"],
        dateStart=f"{day_before}",
        dateEnd=f"{today}",
        lang="eng",
        sourceUri=er.getSourceUri("bbc"),
    )
    articles = []
    titles = []
    images = []
    images_number = []

    for art in query.execQuery(er, sortBy="rel", maxItems=50):
        titles.append(str(art.get("title","")))
        print("titles added")
        articles.append(str(art.get("body", "")))
        images.append(art.get("image", ""))
    for i, img in enumerate(images):
        print("checking")
        try:
            res = requests.get(img,timeout=15)
            content_type = res.headers.get("Content-Type")
            print(content_type)

            if content_type == "image/png":
                file_extension = ".png"
            elif content_type is None:
                images_number.append("images/default_image.png")
                continue
            else:
                file_extension = ".jpg"
            print("checking images")
            with open(f'images/img{i}{file_extension}', 'wb') as f:
                f.write(res.content)
                images_number.append(f"images/img{i}")
        except:
            images_number.append("images/default_image.png")
    dataframe_of_news = pd.DataFrame({
        "titles": titles,
        "articles": articles,
        "images" : images_number
    })
    print("got news")
    return dataframe_of_news