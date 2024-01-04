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


def com_recursion(n, m):
    """
    求n个中取m个数的所有组合,递归
    """
    q = []

    def dfs(i):
        if len(q) > m or len(q) + n - i + 1 < m:
            return
        if i == n + 1:
            print(q)
            return
        dfs(i + 1)
        q.append(i)
        dfs(i + 1)
        q.pop()
        return

    dfs(1)


def com_iteration(n, m):
    def low_bit(temp):
        return (-temp) & temp

    # 从所有的n位二进制数中选出1的个数为m的数字
    # 作为选择的组合的下标
    res_total = []
    for i in range(1 << n):
        kk = i
        sum_k = 0
        while kk:
            kk -= low_bit(kk)  # 移除最低位的1
            sum_k += 1  # 计数
        if sum_k == m:
            res = []
            for j in range(n):
                if i & (1 << j):
                    res.append(j + 1)
            res_total.append(res)
    return res_total


def pop_count(num):
    temp = num  # 求num中1的个数
    temp = (temp & 0x55555555) + ((temp >> 1) & 0x55555555)  # 计算每两位的1的个数,并保存在这两位中
    temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333)  # 将刚才的计算结果每2个一组,组成4位求和,保存在这四位中
    temp = (temp & 0x0F0F0F0F) + ((temp >> 4) & 0x0F0F0F0F)  # 同上 重复
    temp = (temp & 0x00FF00FF) + ((temp >> 8) & 0x00FF00FF)
    temp = (temp & 0x0000FFFF) + ((temp >> 16) & 0x0000FFFF)
    return temp


def count_trailing_zeros(x):
    # 计算尾部0的个数
    return (x & -x).bit_length() - 1
    # 1001
    # 0111
    # 0001
    # return 1-1 = 0


def gospers_hack(k, n):
    # 将最后一个01变成10,然后把它右边的1全部集中到最右边即可
    cur = (1 << k) - 1  # 刚好有k个1,属于一种情况
    limit = (1 << n)
    while cur < limit:
        print(bin(cur))
        lb = cur & -cur  # 取最低位的1
        r = cur + lb  # 在cur最低位的1上加1
        cur = ((r ^ cur) >> count_trailing_zeros(lb) + 2) | r
