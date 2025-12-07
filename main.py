import numpy as np

import Task_1 as t1
import Task_2 as t2

def main():
    # # task 1
    # A = np.array([[1, -1, 1], [-2, 2, -2]])
    # U, E, V = t1.svd(A)

    # task 2
    # part 1
    t2.task_2()

    # part 2
    t2.task_2(50, 50, 3, True)
    t2.task_2(50, 50, 10, True)

    return 0

if __name__ == '__main__':
    main()