from typing import *


def solution_100191(word: str) -> int:
    """
    index : 3014
    """
    if len(word) <= 8:
        return len(word)
    if len(word) <= 16:
        return (len(word) - 8) * 2 + 8
    if len(word) <= 24:
        return (len(word) - 16) * 3 + 24
    return (len(word) - 24) * 4 + 48


def solution_100188(n: int, x: int, y: int) -> List[int]:
    """
    index:3015
    """
    # floyd
    # 超时
    f = [[10 ** 18] * n for i in range(n)]
    f[x - 1][y - 1], f[y - 1][x - 1] = 1, 1
    for i in range(n):
        f[i][i] = 0
    for i in range(n - 1):
        f[i][i + 1], f[i + 1][i] = 1, 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                f[i][j] = min(f[i][j], (f[i][k] + f[k][j]))
    res = [0] * n
    for i in range(n):
        for j in range(n):
            res[f[i][j] - 1] += 1
    res[-1] = 0
    return res


def solution_100188_2(n: int, x: int, y: int) -> List[int]:
    # 暴力做法:BFS
    # 每个点花费O(n)求出它到其余点的距离
    # 花费O(n^2)时间求出所有结果
    res = [0] * n
    x = x - 1
    y = y - 1
    # 只往后找,因此每次结果加+2
    for i in range(n):
        for j in range(i + 1, n):
            d1 = j - i  # 直接走
            d2 = abs(i - x) + 1 + abs(j - y)  # i->x->y->j
            d3 = abs(i - y) + 1 + abs(j - x)  # i->y->x->j
            min_d = min(d1, d2, d3)
            res[min_d - 1] += 2
    return res


def solution_100192(word: str) -> int:
    """
    index: 3016
    """
    f = [0] * 26
    for ch in word:
        f[ord(ch) - ord('a')] += 1
    f.sort(reverse=True)
    res = 0
    res += sum(f[0:8])
    res += sum(f[8:16] * 2)
    res += sum(f[16:24] * 3)
    res += sum(f[24:] * 4)
    return res


def solution_100192_2(word: str) -> int:
    # 排序不等式
    cnt = Counter[str](word)
    a = sorted(cnt.values(), reverse=True)
    ans = 0
    for i, c in enumerate(a):
        ans += c * (i // 8 + 1)
    return ans
