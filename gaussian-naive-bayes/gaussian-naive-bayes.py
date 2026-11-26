import math
from collections import defaultdict

def gaussian_naive_bayes(X_train, y_train, X_test):
    classes = sorted(set(y_train))
    n = len(y_train)
    eps = 1e-9

    stats = {}

    for c in classes:
        rows = [X_train[i] for i in range(n) if y_train[i] == c]
        count = len(rows)
        features = len(X_train[0])

        means = []
        variances = []

        for j in range(features):
            vals = [row[j] for row in rows]
            mean = sum(vals) / count
            var = sum((x - mean) ** 2 for x in vals) / count

            means.append(mean)
            variances.append(var + eps)

        prior = count / n
        stats[c] = (prior, means, variances)

    result = []

    for x in X_test:
        best_class = None
        best_score = float('-inf')

        for c in classes:
            prior, means, variances = stats[c]
            score = math.log(prior)

            for j in range(len(x)):
                var = variances[j]
                mean = means[j]

                score += -0.5 * math.log(2 * math.pi * var)
                score += -((x[j] - mean) ** 2) / (2 * var)

            if score > best_score:
                best_score = score
                best_class = c

        result.append(int(best_class))

    return result