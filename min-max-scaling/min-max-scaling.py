
def min_max_scaling(data):
    rows = len(data)
    cols = len(data[0])

    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):
        col_min = min(data[i][j] for i in range(rows))
        col_max = max(data[i][j] for i in range(rows))

        for i in range(rows):
            if col_max == col_min:
                result[i][j] = 0.0
            else:
                result[i][j] = (
                    data[i][j] - col_min
                ) / (col_max - col_min)

    return result
