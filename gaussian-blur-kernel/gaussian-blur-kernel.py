import math

def gaussian_kernel(size, sigma):
    center = size // 2
    kernel = [[0.0] * size for _ in range(size)]
    total = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center

            weight = math.exp(-(x*x + y*y) / (2 * sigma * sigma))
            kernel[i][j] = weight
            total += weight

    for i in range(size):
        for j in range(size):
            kernel[i][j] /= total

    return kernel