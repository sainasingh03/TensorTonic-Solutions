def precision_recall_at_k(recommended, relevant, k):
    relevant_set = set(relevant)

    hits = sum(1 for x in recommended[:k] if x in relevant_set)

    precision = hits / k
    recall = hits / len(relevant)

    return [precision, recall]