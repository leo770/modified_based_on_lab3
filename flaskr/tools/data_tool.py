import os
import pandas as pd


def loadData():
    return getMovies(), getGenre(), getRates()


# movieId,title,year,overview,cover_url,genres
def getMovies():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/data_project/books_metadata_large.csv"
    # df = pd.read_csv(path, delimiter=",", names=["movieId", "title", "year", "overview", "cover_url", "genres"])
    df = pd.read_csv(path)
    df.set_index('book_id')

    return df


# A list of the genres.
def getGenre():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/data_project/books_metadata_large.csv"
    df = pd.read_csv(path)
    df_select = df[["format", "book_id"]]
    df_select.set_index('book_id')
    return df_select


# user id | book id | rating | timestamp
def getRates():
    rootPath = os.path.abspath(os.getcwd())
    path = f"{rootPath}/flaskr/static/data_project/ratings.csv"
    df = pd.read_csv(path, delimiter=",", names=["userId", "bookId", "rating", "timestamp"])
    df = df.drop(columns='timestamp')
    df = df[['userId', 'bookId', 'rating']]

    return df



# itemID | userID | rating
def ratesFromUser(rates):
    itemID = []
    userID = []
    rating = []

    for rate in rates:
        items = rate.split("|")
        userID.append(int(items[0]))
        itemID.append(int(items[1]))
        rating.append(int(items[2]))

    ratings_dict = {
        "userId": userID,
        "bookId": itemID,
        "rating": rating,
    }

    return pd.DataFrame(ratings_dict)