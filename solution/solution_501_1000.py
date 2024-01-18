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
