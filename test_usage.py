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


def mul_simple(x, y):
    # x = [..,..,..,..]
    return [x[0] * y[0] + x[1] * y[2],
            x[0] * y[1] + x[0] * y[3],
            x[2] * y[0] + x[3] * y[2],
            x[2] * y[1] + x[3] * y[3]]

if __name__ == '__main__':
    x = [[1,1],[1,0]]
    print(matrix_mul(x,x))
    print(mul_simple([1,1,1,0],[1,1,1,0]))
    print(len(x[0]))
    print(x[1][1])