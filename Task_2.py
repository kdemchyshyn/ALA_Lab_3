import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-v0_8")

def plot_data(mat, name, camera=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(name)

    if camera != None:
        ax.view_init(elev=camera[0], azim=camera[1])
    for index, row in mat.iterrows():
        ax.scatter(row[0], row[1], row[2], alpha=0.8)
        ax.text(row[0], row [1], row[2], '', size=10)
    plt.show()

# part 1
# Reading the CSV file
file_path = '.\\database\\ratings.csv'
df = pd.read_csv(file_path)
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')
# filter useful data
ratings_matrix = ratings_matrix.dropna(thresh=200, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=60, axis=1)
print (ratings_matrix)

# normalize data
ratings_matrix_filled = ratings_matrix.fillna(2.5) # hehehe product analytics: avg, mode, 0 or -1
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# svd + visualize
U, sigma, Vt = svds(R_demeaned, k=3)
plot_data(pd.DataFrame(U[:15]), "Users (U)")
plot_data(pd.DataFrame(Vt.T[:15]), "Movies (V)")

# part 2


