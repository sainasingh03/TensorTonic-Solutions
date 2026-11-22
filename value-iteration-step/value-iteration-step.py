def value_iteration_step(values, transitions, rewards, gamma):
    n = len(values)
    ans = []

    for s in range(n):
        best = float('-inf')

        for a in range(len(rewards[s])):
            q = rewards[s][a]

            for nxt in range(n):
                q += gamma * transitions[s][a][nxt] * values[nxt]

            best = max(best, q)

        ans.append(float(best))

    return ans