
import math

def elu(x, alpha):
    result = []

    for value in x:
        if value > 0:
            result.append(float(value))
        else:
            result.append(alpha * (math.exp(value) - 1))

    return result
