import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-v0_8")

def plot_data(mat, name, camera=None):
    mat = pd.DataFrame(mat)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(name)

    if camera != None:
        ax.view_init(elev=camera[0], azim=camera[1])
    for index, row in mat.iterrows():
        ax.scatter(row[0], row[1], row[2], alpha=0.8)
        ax.text(row[0], row [1], row[2], '{0}'.format(index), size=10)
    plt.show()

def recommend(user, matrix, output_num=10):
    movies = matrix[matrix.index == user]
    rec = []
    for item in movies.columns:
        if (not movies[item].isna().any()):
            rec.append([item, matrix.loc[user, item]])
    fin_rec = [i[0] for i in sorted(rec, key=lambda x: x[1],reverse=True)]
    fin_rec = fin_rec[:output_num]

    file_path = '.\\database\\movies.csv'
    movies_df = pd.read_csv(file_path)
    print (f"User {user}: ")
    print ((movies_df[movies_df['movieId'].isin(fin_rec)])[['title', 'genres']], '\n')

def task_2(nan_u=200, nan_v=100, dim=3, part_2=False):
    # part 1
    # Reading the CSV file
    file_path = '.\\database\\ratings.csv'
    df = pd.read_csv(file_path)
    ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

    # filter useful data
    ratings_matrix = ratings_matrix.dropna(thresh=nan_u, axis=0)
    ratings_matrix = ratings_matrix.dropna(thresh=nan_v, axis=1)

    # normalize data
    ratings_matrix_filled = ratings_matrix.fillna(2.5)
    R = ratings_matrix_filled.values
    user_ratings_mean = np.mean(R, axis=1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    # svd + visualize
    U, sigma, Vt = svds(R_demeaned, k=dim)
    sigma = np.diag(sigma)

    if (not part_2):
        plot_data(U[:15], "Users (U)")
        plot_data(Vt.T[:15], "Movies (V)")

    # part 2
    if(part_2):
        print("Start matrix: \n", ratings_matrix)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)
        print("Prediction matrix: \n", preds_df)

        only_predicted = preds_df[ratings_matrix.isna()]
        print("Only predictions matrix: \n", only_predicted)

        users = [1]
        for user in users:
            recommend(user, only_predicted)
