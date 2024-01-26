from functools import cache
from typing import *


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
    # 100100
    # 011011
    # 011100
    # return 1-1 = 0


def count_trailing_zeros_2(x):
    c = 31
    x = x & -x
    if x & 0x0000ffff:
        c -= 16
    if x & 0x00ffffff:
        c -= 8
    if x & 0x0f0f0f0f:
        c -= 4
    if x & 0x33333333:
        c -= 2
    if x & 0x55555555:
        c -= 1
    return c


def count_leading_zeros(x):
    clz = 0  # 前导0的数量,用于计算num的二进制长度
    if x >> 16 == 0:
        clz += 16
        x = x << 16
    if x >> 24 == 0:
        clz += 8
        x = x << 8
    if x >> 28 == 0:
        clz += 4
        x = x << 4
    if x >> 30 == 0:
        clz += 2
        x = x << 2
    if x >> 31 == 0:
        clz += 1


def gospers_hack(k, n) -> List[int]:
    # 将最后一个01变成10,然后把它右边的1全部集中到最右边即可
    res = []
    cur = (1 << k) - 1  # 刚好有k个1,属于一种情况
    limit = (1 << n)
    while cur < limit:
        res.append(cur)
        lb = cur & -cur  # 取最低位的1
        r = cur + lb  # 在cur最低位的1上加1
        cur = ((r ^ cur) >> count_trailing_zeros(lb) + 2) | r
    return res


def gcd_euclid(a: int, b: int) -> int:
    """
    Compute the greatest common
    """
    while a % b != 0:
        a, b = b, a % b
    return b


def countSpecialNumbers(n: int) -> int:
    """
    数位DP模板
    """
    s = str(n)

    @cache
    def f(i: int, mask: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return int(is_num)  # is_num 为 True 表示得到了一个合法数字
        res = 0
        if not is_num:
            res = f(i + 1, mask, False, False)  # 当前位置没填数字，且前面也没填数字
        low = 0 if is_num else 1  # 如果前面填了数字，这里就从0开始，否则这一位至少是1，这一位也不填的结果有上面一行得到，下面只用考虑1-9的情况
        up = int(s[i]) if is_limit else 9
        for d in range(low, up + 1):
            if (mask >> d & 1) == 0:
                res += f(i + 1, mask | (1 << d), is_limit and d == up, True)  # 当前位置填数字了，钱吗
        return res

    return f(0, 0, True, False)


def quick_sort(nums, left, right):
    if left >= right:
        return
    i = left
    j = right
    temp = nums[left]
    while i < j:
        # 因为中枢元素是nums[left],所以空洞先出现在左侧
        while i < j and nums[j] >= temp:
            j -= 1
        # 不会产生覆盖
        nums[i] = nums[j]
        while i < j and nums[i] <= temp:
            i += 1
        nums[j] = nums[i]
    nums[i] = temp
    # 中枢元素不需要再修改
    quick_sort(nums, left, i - 1)
    quick_sort(nums, i + 1, right)


def lca_simple(edges: List[List[int]], node1: int, node2: int) -> int:
    n = len(edges) + 1  # 节点个数
    depth = [0] * n
    parent = [-1] * n
    g = [[] for _ in range(n)]  # 存储每个节点所持有的边
    for x, y in edges:
        g[x].append(y)
        g[y].append(x)

    def dfs(node: int, p: int):
        parent[node] = p
        depth[node] = depth[p] + 1
        for y in g[node]:
            if y != p:
                dfs(y, node)

    dfs(0, -1)  # 标记所有节点的深度
    while node1 != node2:
        if depth[node1] > depth[node2]:
            node1 = parent[node1]
        else:
            node2 = parent[node2]
    return node1


def lca_bin_lift(edges: List[List[int]], node1: int, node2: int) -> int:
    n = len(edges) + 1  # 节点个数
    m = n.bit_length()  # 2倍幂上升的最大次数
    depth = [0] * n

    g = [[] for _ in range(n)]  # 存储每个节点所持有的边
    for x, y in edges:
        g[x].append(y)
        g[y].append(x)

    parent = [[-1] * m for _ in range(n)]

    # parent[x][0]表示x的父节点，
    # parent[x][1] = parent[x][0][0] 表示x的父节点的父节点
    # parent[x][2] = parent[x][1][1] 表示x向上四步的节点
    # 2倍幂的上跳速度
    def dfs(node: int, p: int):
        parent[node][0] = p
        depth[node] = depth[p] + 1
        for y in g[node]:
            if y != p:
                dfs(y, node)

    dfs(0, -1)  # 标记所有节点的深度
    for i in range(m - 1):
        for x in range(n):
            if (p := parent[x][i]) != -1:
                parent[x][i + 1] = parent[p][i]  # 更新2倍幂上跳对应的节点

    def get_kth(node: int, k: int) -> int:
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                node = parent[node][i]
                if node < 0:
                    break
        return node

    # 以上为预处理，只需要做1遍

    if depth[node1] > depth[node2]:
        node1, node2 = node2, node1
    node2 = get_kth(node2, depth[node2] - depth[node1])  # 使得node2和node1位于同一深度
    if node1 == node2:
        return node1
    # 贪心上跳
    for i in range(len(parent[node2]) - 1, -1, -1):
        p1, p2 = parent[node1][i], parent[node2][i]
        if p1 != p2:  # 还能继续往上跳
            node1, node2 = p1, p2
    # 为什么是parent[node1][0]，也就是当前node的父节点
    # 贪心上跳的for循环没有提前结束的可能
    # 因此一定会走完全部logn层，也就是会尝试所有可能的高度
    # 一定会到达lca的某2个子节点，且不相同
    # 此次无论步长是多少，都无法再更新node1和node2了
    # 所以结果是parent[node1][0]
    return parent[node1][0]


class TreeAncestorTemplate:
    def __init__(self, edges: List[List[int]]):
        n = len(edges) + 1
        m = n.bit_length()
        depth = [0] * n
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        parent = [[-1] * m for _ in range(n)]

        def dfs(x, p):
            parent[x][0] = p
            depth[x] = depth[p] + 1
            for y in g[x]:
                if y != p:
                    dfs(y, x)

        dfs(0, -1)
        for i in range(m - 1):
            for x in range(n):
                if (p := parent[x][i]) != -1:
                    parent[x][i + 1] = parent[p][i]
        self.parent = parent
        self.depth = depth

    def getKthAncestor(self, node: int, k: int) -> int:
        res = node
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                res = self.parent[res][i]
        return res


class TreeAncestor:
    def __init__(self, n: int, parent: List[int]):
        m = n.bit_length()
        pa = [[-1] * m for _ in range(n)]
        for i in range(n):
            pa[i][0] = parent[i]
        for i in range(m - 1):
            for x in range(n):
                if (p := pa[x][i]) != -1:
                    pa[x][i + 1] = pa[p][i]
        self.pa = pa

    def getKthAncestor(self, node: int, k: int) -> int:
        res = node
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                res = self.pa[res][i]
                if res < 0:
                    break
        return res
