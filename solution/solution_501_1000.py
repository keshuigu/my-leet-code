from typing import *
from .data_struct import *


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
