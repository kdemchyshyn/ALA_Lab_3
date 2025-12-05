import numpy as np

import Task_1 as t1
import Task_2 as t2

def main():
    # task 1
    A = np.array([[1, 1], [0, 2]])
    t1.svd(A)
    return 0

if __name__ == '__main__':
    main()