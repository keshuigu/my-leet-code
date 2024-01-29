from typing import *
from .data_struct import *
from functools import cache


def solution_876(head: Optional[ListNode]) -> Optional[ListNode]:
    # 快慢指针
    p = q = head
    while p is not None and p.next is not None:
        p = p.next.next
        q = q.next
    return q


def solution_600(n: int) -> int:
    s = bin(n)[2:]
    length = len(s)
    dp = [[-1] * 2 for _ in range(length)]

    def f(i, j, limit):
        if i == length:
            return 1
        if not limit and dp[i][j] != -1:
            return dp[i][j]
        res = 0
        # 复杂
        if j == 1:
            res += f(i + 1, 0, limit and s[i] == '0')
        else:
            res += f(i + 1, 0, limit and s[i] == '0')
            if s[i] == '1' or not limit:
                res += f(i + 1, 1, limit and s[i] == '1')
        # up = int(s[i]) if limit else 1
        # res += f(i + 1, 0, limit and up == 0)
        # if j == 0 and up == 1:
        #     res += f(i + 1, 1, limit)
        if not limit:
            dp[i][j] = res
        return res

    return f(0, 0, True)


def solution_902(digits: List[str], n: int) -> int:
    s = str(n)
    ld = len(digits)
    ls = len(s)
    dp = [-1] * ls
    my_set = set(digits)
    min_num = int(digits[0])
    max_num = int(digits[ld - 1])

    def dfs(i, limit, num):
        if i == ls:
            return 1 if num else 0
        if not limit and dp[i] != -1:
            return dp[i]
        res = 0
        if not num:
            res += dfs(i + 1, False, False)
        low = min_num
        up = int(s[i]) if limit else max_num
        for d in range(low, up + 1):
            if str(d) in my_set:
                res += dfs(i + 1, limit and d == up, True)
        if not limit and num:
            dp[i] = res
        return res

    return dfs(0, True, False)


def solution_670(num: int) -> int:
    s = str(num)
    chs = list(enumerate(s))
    chs.sort(reverse=True, key=lambda x: x[1])
    for i in range(len(chs)):
        index, ch = chs[i]
        if ch != s[i]:
            for j in range(i + 1, len(chs)):
                if chs[j][1] == ch:
                    index = chs[j][0]
                else:
                    break
            res = s[:i] + ch + s[i + 1:index] + s[i] + s[index + 1:]
            return int(res)
    return num


def solution_670_2(num: int) -> int:
    s = list(str(num))
    max_idx = len(s) - 1
    p = q = -1
    for i in range(len(s) - 2, -1, -1):
        if s[i] > s[max_idx]:
            max_idx = i
        elif s[i] < s[max_idx]:
            p, q = i, max_idx
    if p == -1:
        return num
    s[p], s[q] = s[q], s[p]
    return int(''.join(s))


def solution_514(ring: str, key: str) -> int:
    s = [ord(c) - ord('a') for c in ring]
    t = [ord(c) - ord('a') for c in key]
    n = len(s)
    # 先算出每个字母的最后一次出现的下标
    pos = [0] * 26
    for i, c in enumerate(s):
        pos[c] = i
    # 计算每个s[i]左边a-z的最近下标
    # pos保存的是从右边数的最后一个下标，因此也就是从左边数的第一个下标
    # 上面的循环结束刚好是s[0]对应的所有字母从左数的第一个下标
    # 更新pos[c]相当于更新这个环的起始位置，更新对于s[i+1]来言，最近的s[i]所在的位置
    # 其他字母的位置不会变化
    left = [None] * n
    for i, c in enumerate(s):
        left[i] = pos[:]
        pos[c] = i

    # 对右边的计算类似,需要倒序来计算最早出现的下标
    for i in range(n - 1, -1, -1):
        pos[s[i]] = i
    right = [None] * n
    for i in range(n - 1, -1, -1):
        right[i] = pos[:]
        pos[s[i]] = i

    # 对于当前s[i],旋转到t[j]所需要的最小开销
    @cache
    def dfs(i: int, j: int) -> int:
        if j == len(t):
            return 0
        c = t[j]
        if s[i] == c:
            return dfs(i, j + 1)
        # 左侧还是右侧最小值
        l, r = left[i][c], right[i][c]
        return min(dfs(l, j + 1) + ((n - l + i) if l > i else i - l),
                   dfs(r, j + 1) + ((n - i + r) if r < i else r - i))

    return dfs(0, 0) + len(t)


def solution_514_2(ring: str, key: str) -> int:
    s = [ord(c) - ord('a') for c in ring]
    t = [ord(c) - ord('a') for c in key]
    n = len(s)
    # 先算出每个字母的最后一次出现的下标
    pos = [0] * 26
    for i, c in enumerate(s):
        pos[c] = i
    # 计算每个s[i]左边a-z的最近下标
    # pos保存的是从右边数的最后一个下标，因此也就是从左边数的第一个下标
    # 上面的循环结束刚好是s[0]对应的所有字母从左数的第一个下标
    # 更新pos[c]相当于更新这个环的起始位置，更新对于s[i+1]来言，最近的s[i]所在的位置
    # 其他字母的位置不会变化
    left = [None] * n
    for i, c in enumerate(s):
        left[i] = pos[:]
        pos[c] = i

    # 对右边的计算类似,需要倒序来计算最早出现的下标
    for i in range(n - 1, -1, -1):
        pos[s[i]] = i
    right = [None] * n
    for i in range(n - 1, -1, -1):
        right[i] = pos[:]
        pos[s[i]] = i

    # 对于当前s[i],旋转到t[j]所需要的最小开销
    memo = [[-1] * n for _ in range(len(t))]

    def dfs(i: int, j: int) -> int:
        if j == len(t):
            return 0
        if memo[j][i] != -1:
            return memo[j][i]
        c = t[j]
        if s[i] == c:
            memo[j][i] = dfs(i, j + 1)
            return memo[j][i]
        # 左侧还是右侧最小值
        l, r = left[i][c], right[i][c]
        memo[j][i] = min(dfs(l, j + 1) + ((n - l + i) if l > i else i - l),
                         dfs(r, j + 1) + ((n - i + r) if r < i else r - i))
        return memo[j][i]

    return dfs(0, 0) + len(t)
