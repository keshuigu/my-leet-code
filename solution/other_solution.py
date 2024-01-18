def solution_iq_17_06(n: int) -> int:
    s = str(n)
    length = len(s)
    dp = [[-1] * length for _ in range(length)]

    def f(i, j, limit):
        if i == length:
            return j
        if not limit and dp[i][j] != -1:
            return dp[i][j]
        res = 0
        up = int(s[i]) if limit else 9
        for d in range(up + 1):
            res += f(i + 1, j + (d == 2), limit and d == up)
        if not limit:
            dp[i][j] = res
        return res

    return f(0, 0, True)
