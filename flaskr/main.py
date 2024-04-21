from flask import (
    Blueprint, render_template, request
)

from .tools.data_tool import *

from surprise import Reader
from surprise import KNNBasic
from surprise import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from surprise.prediction_algorithms.matrix_factorization import SVD

bp = Blueprint('main', __name__, url_prefix='/')

movies, genres, rates = loadData()


@bp.route('/', methods=('GET', 'POST'))
def index():

    # Default Genres List
    default_genres = genres.to_dict('records')[:3] #first item page to modify

    # User Genres
    user_genres = request.cookies.get('user_genres')
    if user_genres:
        user_genres = user_genres.split(",")
    else:
        user_genres = []

    # User Rates
    user_rates = request.cookies.get('user_rates')
    if user_rates:
        user_rates = user_rates.split(",")
    else:
        user_rates = []

    # User Likes
    user_likes = request.cookies.get('user_likes')
    if user_likes:
        user_likes = user_likes.split(",")
    else:
        user_likes = []

    default_bookings = getbookings(user_genres)[:12]
    recommendations_movies, recommendations_message = getRecommendationBy(user_rates)
    likes_similar_movies, likes_similar_message = getLikedSimilarBy([int(numeric_string) for numeric_string in user_likes])
    likes_movies = getUserLikesBy(user_likes)

    return render_template('index.html',
                           genres=default_genres,
                           user_genres=user_genres,
                           user_rates=user_rates,
                           user_likes=user_likes,
                           default_bookings=default_bookings,
                           recommendations=recommendations_movies,
                           recommendations_message=recommendations_message,
                           likes_similars=likes_similar_movies,
                           likes_similar_message=likes_similar_message,
                           likes=likes_movies,
                           )


def getUserLikesBy(user_likes):
    results = []

    if len(user_likes) > 0:
        mask = movies['book_id'].isin([int(book_id) for book_id in user_likes])
        results = movies.loc[mask]

        original_orders = pd.DataFrame()
        for _id in user_likes:
            book = results.loc[results['book_id'] == int(_id)]
            if len(original_orders) == 0:
                original_orders = book
            else:
                original_orders = pd.concat([book, original_orders])
        results = original_orders

    # return the result
    if len(results) > 0:
        return results.to_dict('records')
    return results


# def getMoviesByGenres(user_genres):
#     results = []

#     # ====  Do some operations ====

#     if len(user_genres) > 0:
#         genres_mask = genres['book_id'].isin([int(book_id) for book_id in user_genres])
#         user_genres = [1 if has is True else 0 for has in genres_mask]
#         user_genres_df = pd.DataFrame(user_genres)
#         user_genres_df.index = genres['format']
#         movies_genres = movies.iloc[:, 5:]
#         mask = (movies_genres.dot(user_genres_df) > 0).squeeze()
#         results = movies.loc[mask][:30]

#     # ==== End ====

#     # return the result
#     if len(results) > 0:
#         return results.to_dict('records')
#     return results

def getbookings(user_genres):
    results = []

    # ====  Do some operations ====

    if len(user_genres) > 0:
        results = movies

    # ==== End ====

    # return the result
    if len(results) > 0:
        return results.to_dict('records')
    return results


def getRecommendationBy(user_rates):
    results = []

    # Check if there are any user_rates
    if len(user_rates) > 0:
        # Initialize a reader with rating scale from 1 to 5
        reader = Reader(rating_scale=(1, 5))

        # Convert user_rates to rates from the user
        user_rates = ratesFromUser(user_rates)

        # Combine rates and user_rates into training_rates
        training_rates = pd.concat([rates, user_rates], ignore_index=True)

        # Load the training data from the training_rates DataFrame
        training_data = Dataset.load_from_df(training_rates, reader=reader)

        # Split the data into training and testing sets
        trainset = training_data.build_full_trainset()

        # User-based Collaborative Filtering (KNN)
        algo_knn = KNNBasic(sim_options={'name': 'pearson', 'user_based': True})
        algo_knn.fit(trainset)

        # SVD Matrix Factorization
        algo_svd = SVD()
        algo_svd.fit(trainset)

        # Convert the raw user id to the inner user id using trainset
        inner_id = trainset.to_inner_uid(611)

        # Get the nearest neighbors of the inner_id using KNN
        neighbors = algo_knn.get_neighbors(inner_id, k=1)
        neighbors_uid = [trainset.to_raw_uid(x) for x in neighbors]

        # Calculate weights based on the number of neighbors
        knn_weight = len(neighbors) / len(trainset.all_users())

        # Filter out books liked by neighbors from KNN
        knn_books = rates[rates['userId'].isin(neighbors_uid)]
        knn_books_ids = knn_books[knn_books['rating'] > 2.5]['bookId']

        # Use SVD to recommend books not rated by the user but liked by neighbors
        svd_books_ids = []
        for book_id in knn_books_ids:
            if book_id not in user_rates['bookId'].values:
                prediction = algo_svd.predict(611, book_id)
                if prediction.est > 3.0:  # Adjust threshold as needed
                    svd_books_ids.append(book_id)

        # Calculate weights for SVD recommendations
        svd_weight = 1.0 - knn_weight

        # Combine recommendations from KNN and SVD with weights
        combined_books_ids = knn_books_ids.tolist() + svd_books_ids
        combined_books_weights = [knn_weight] * len(knn_books_ids) + [svd_weight] * len(svd_books_ids)

        # Get details of the combined books
        combined_books = movies[movies['book_id'].isin(combined_books_ids)]

        # Create a DataFrame for the combined recommendations
        combined_recommendations = pd.DataFrame({
            'book_id': combined_books_ids,
            'weight': combined_books_weights
        })

        # Merge combined_books with combined_recommendations on 'book_id' to include weights
        combined_books = pd.merge(combined_books, combined_recommendations, on='book_id')

        # Sort the combined books by weight in descending order
        combined_books = combined_books.sort_values(by='weight', ascending=False).head(12)

        # Extract the final recommended books
        results = combined_books.drop(columns='weight')

    # Return the result
    if len(results) > 0:
        return results.to_dict('records'), "These books are recommended based on your ratings and hybrid recommendation."
    return results, "No recommendations."


# Modify this function
def getLikedSimilarBy(user_likes):
    results = []

    # ==== Do some operations ====
    if len(user_likes) > 0:

        # Step 1: Representing items with one-hot vectors
        item_rep_matrix, item_rep_vector, feature_list = item_representation_based_book_types(movies)

        # Step 2: Building user profile
        user_profile = build_user_profile(user_likes, item_rep_vector, feature_list)

        # Step 3: Predicting user interest in items
        results = generate_recommendation_results(user_profile, item_rep_matrix, item_rep_vector, 12)

    # Return the result
    if len(results) > 0:
        return results.to_dict('records'), "The books are similar to your liked books."
    return results, "No similar books found."

    # ==== End ====


def item_representation_based_book_types(books_df):
    books_with_types = books_df.copy(deep=True)

    genre_list = books_with_types.columns[5:]
    movies_genre_matrix = books_with_types[genre_list].to_numpy()
    return movies_genre_matrix, books_with_types, genre_list


def build_user_profile(bookIds, item_rep_vector, feature_list, normalized=True):

    ## Calculate item representation matrix to represent user profiles
    user_book_rating_df = item_rep_vector[item_rep_vector['book_id'].isin(bookIds)]
    user_book_df = user_book_rating_df[feature_list].mean()
    user_profile = user_book_df.T

    if normalized:
        user_profile = user_profile / sum(user_profile.values)

    return user_profile


def generate_recommendation_results(user_profile,item_rep_matrix, movies_data, k=12):

    u_v = user_profile.values
    u_v_matrix = [u_v]

    # Comput the cosine similarity
    print("u_v_matrix:", u_v_matrix)
    print("item_rep_matrix:", item_rep_matrix)
    recommendation_table = cosine_similarity(u_v_matrix, item_rep_matrix)

    recommendation_table_df = movies_data.copy(deep=True)
    recommendation_table_df['similarity'] = recommendation_table[0]
    rec_result = recommendation_table_df.sort_values(by=['similarity'], ascending=False)[:k]

    return rec_result