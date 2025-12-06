import numpy as np

def printMatrix(X):
    for element in X:
        print(element)

def svd(A):
    E = np.zeros(shape=A.shape) # sigma matrix

    # find AAt and AtA
    left_matrix = np.dot(A, A.T)
    right_matrix = np.dot(A.T, A)

    print("Left matrix: ")
    printMatrix(left_matrix)
    print("Right matrix: ")
    printMatrix(right_matrix)

    # find eigenvalues and vectors
    eigenvalues_v, eigenvectors_v = np.linalg.eig(right_matrix)
    indexes = np.argsort(eigenvalues_v)[::-1]
    eigenvalues_v = eigenvalues_v[indexes]
    eigenvectors_v = eigenvectors_v[:, indexes]

    eigenvalues_u, eigenvectors_u = np.linalg.eig(left_matrix)
    indexes = np.argsort(eigenvalues_u)[::-1]
    eigenvalues_u = eigenvalues_u[indexes]
    eigenvectors_u = eigenvectors_u[:, indexes]

    # find singulars
    singular_nums = np.sqrt(eigenvalues_u) if len(eigenvalues_u) < len(eigenvalues_v) else np.sqrt(eigenvalues_v)

    # construct matrixes
    for i in range(len(singular_nums)):
        E[i, i] = singular_nums[i]

    print("Matrix E:")
    printMatrix(E)

    V = eigenvectors_v
    U = eigenvectors_u

    for i in range(len(singular_nums)):
        if singular_nums[i] != 0:
            U[:, i] = np.dot(A, V[:, i]) / singular_nums[i]

    print("Matrix U:")
    printMatrix(U)
    print("Matrix V:")
    printMatrix(V)

    A_test = np.dot(np.dot(U, E), V.T)
    if not np.allclose(A, A_test):
        print("Matrixes not equal, wrong svd")
        print("Original matrix: ")
        printMatrix(A)
        print("UEVt matrix: ")
        printMatrix(A_test)
        return None, None, None

    print("Original matrix: ")
    printMatrix(A)
    print("UEVt matrix: ")
    printMatrix(A_test)

    return U, E, V