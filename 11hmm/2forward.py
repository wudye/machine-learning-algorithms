def forward_algorithm(a, b, p, sequence):
    n_state = len(a)

    dp = [p[i] * b[i][sequence[0]] for i in range(n_state)]
    for k in range (1, len(sequence)):
        dp = [sum(a[j][i] * dp[j] for j in range(n_state)) * b[i][sequence[k]] for i in range(n_state)]
        """
        new_dp = [0.0] * n_state 
        for i in range(n_state):
            transition_sum = 0.0
            for j in range(n_state):
                transition_sum += dp[j] * a[j][i]
            new_dp[i] = transition_sum * b[i][sequence[k]]
        dp = new_dp 
        """
    return sum(dp)

if __name__ == "__main__":
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]
    p = [0.2, 0.4, 0.4]
    print(forward_algorithm(A, B, p, [0, 1, 0]))  # 0.130218