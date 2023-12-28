def matrix_mul(x, y):
    """
    计算矩阵的乘积,未考虑异常
    :param x: p*q的矩阵,用列表表示
    :param y: q*m的矩阵,用列表表示
    :return: p*m的矩阵
    """
    p = len(x)
    q = len(x[0])
    m = len(y[0])
    res = list(range(p))
    for i in range(p):
        res[i] = list(range(m))
        for j in range(m):
            temp = 0
            for k in range(q):
                temp = temp + x[i][k] * y[k][j]
            res[i][j] = temp

    return res


def quick_pow(a: int, n: int) -> int:
    ans = 1
    while n > 0:  # 指数不为0
        if n & 1:  # 二进制当前为1,说明结果需要乘以当前底数的2的若干次方
            ans *= a
        a *= a  # 计算下一个底数的2的若干次方
        n = n >> 1
    return ans
