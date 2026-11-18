
import numpy as np

def matrix_transpose(A):
    n = len(A)
    m = len(A[0])

    result = np.zeros((m, n), dtype=np.array(A).dtype)

    for i in range(n):
        for j in range(m):
            result[j][i] = A[i][j]

    return result
