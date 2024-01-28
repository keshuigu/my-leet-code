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


def solution_100215(s: str) -> int:
    tmp = s.lower()
    cnt = 0
    for i in range(len(tmp) - 1):
        if tmp[i] != tmp[i + 1]:
            cnt += 1
    return cnt


def solution_100206(nums: List[int]) -> int:
    f = {}
    for num in nums:
        if num not in f:
            f[num] = 1
        else:
            f[num] += 1
    max_cnt = 1
    for num in nums:
        cnt = 2
        if num == 1 and f[num] % 2 == 0:
            cnt = f[num] - 1
        elif num == 1 and f[num] % 2 != 0:
            cnt = f[num]
        elif f[num] >= 2:
            tmp = num
            while True:
                tmp = tmp * tmp
                if tmp not in f:
                    cnt -= 1
                    break
                elif f[tmp] == 1:
                    cnt += 1
                    break
                elif f[tmp] >= 2:
                    cnt += 2
        else:
            cnt -= 1
        max_cnt = max(max_cnt, cnt)
    return max_cnt


def solution_100195(n: int, m: int) -> int:
    even_x = (n + 1) // 2
    even_y = (m + 1) // 2
    odd_x = n - even_x
    odd_y = m - even_y
    return even_x * odd_y + even_y * odd_x


def solution_100179(nums: List[int], k: int) -> int:
    ...
