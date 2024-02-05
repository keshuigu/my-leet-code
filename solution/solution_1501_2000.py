import heapq
from collections import deque
from typing import *
from .data_struct import *


def solution_1672(accounts: List[List[int]]) -> int:
    max_account = 0
    for account in accounts:
        one_total = 0
        for bank in account:
            one_total = one_total + bank
        if one_total > max_account:
            max_account = one_total
    return max_account


def solution_1599(customers: List[int], boardingCost: int, runningCost: int) -> int:
    profit = []
    waiting = customers[0]
    turn = 0
    while True:
        board = waiting if waiting < 4 else 4
        old = 0 if turn == 0 else profit[turn - 1]
        profit.append(old + board * boardingCost - runningCost)
        turn += 1
        if turn >= len(customers):
            waiting = waiting - board
        else:
            waiting = waiting - board + customers[turn]
        if waiting == 0 and turn >= len(customers):
            final_profit = max(max(profit), 0)
            if final_profit == 0:
                return -1
            else:
                return profit.index(final_profit) + 1


def solution_1944(heights: List[int]) -> List[int]:
    stack = []
    res = []
    for height in reversed(heights):
        count = 0
        while len(stack) != 0:
            p = stack[-1]
            if p < height:
                stack.pop()
                count += 1
            else:
                stack.append(height)
                break
        if len(stack) == 0:
            stack.append(height)
            res.append(count)
            continue
        else:
            res.append(count + 1)
    return res[::-1]


def solution_1686(aliceValues: List[int], bobValues: List[int]) -> int:
    heap = []
    for i in range(len(aliceValues)):
        heapq.heappush(heap, (-(aliceValues[i] + bobValues[i]), i))
    turn = True
    a_value = 0
    b_value = 0
    while len(heap) > 0:
        _, index = heapq.heappop(heap)
        if turn:
            a_value += aliceValues[index]
        else:
            b_value += bobValues[index]
        turn = not turn
    if a_value == b_value:
        return 0
    elif a_value > b_value:
        return 1
    else:
        return -1


def solution_1686_2(aliceValues: List[int], bobValues: List[int]) -> int:
    # 建立a[i],b[i]的数组,以他们的和排序
    pair = sorted(zip(aliceValues, bobValues), key=lambda p: -p[0] - p[1])
    # alice拿走下标为偶数的数，并加aliceValue
    # bob拿走下标为奇数的数，并加bobValue
    # diff为alice - bob
    diff = sum(x if i % 2 == 0 else -y for i, (x, y) in enumerate(pair))
    return (diff > 0) - (diff < 0)


def solution_1690(stones: List[int]) -> int:
    s = [0]
    for i in range(1, len(stones) + 1):
        s.append(s[i - 1] + stones[i - 1])
    # dfs(i,j)表示问题当前石子为[i,j],最大化自己的得分
    # dp 保存子问题dfs(i,j)
    # 循环不变量 dfs(i,j) = max(s[j + 1] - s[i + 1] - dfs(i + 1, j), s[j] - s[i] - dfs(i, j - 1))
    dp = [[-1] * len(stones) for _ in range(len(stones))]

    def dfs(i, j):
        if i == j:
            return 0
        if dp[i][j] != -1:
            return dp[i][j]
        dp[i][j] = max(s[j + 1] - s[i + 1] - dfs(i + 1, j), s[j] - s[i] - dfs(i, j - 1))
        return dp[i][j]

    return dfs(0, len(stones) - 1)


def solution_1690_2(stones: List[int]) -> int:
    # 转递推
    s = [0]
    for i in range(1, len(stones) + 1):
        s.append(s[i - 1] + stones[i - 1])
    # dfs(i,j)表示问题当前石子为[i,j],最大化自己的得分
    # dp 保存子问题dfs(i,j)
    # 循环不变量 dfs(i,j) = max(s[j + 1] - s[i + 1] - dfs(i + 1, j), s[j] - s[i] - dfs(i, j - 1))
    dp = [[-1] * len(stones) for _ in range(len(stones))]
    for i in range(len(stones) - 2, -1, -1):
        for j in range(1, len(stones)):
            if i == j:
                dp[i][j] = 0
            else:
                dp[i][j] = max(s[j + 1] - s[i + 1] - dp[i + 1][j], s[j] - s[i] - dp[i][j - 1])
    return dp[0][-1]


def solution_1696(nums: List[int], k: int) -> int:
    """
    超时
    """
    n = len(nums)
    dp = [-10 ** 9] * n
    dp[n - 1] = nums[n - 1]

    def dfs(i):
        if dp[i] != -10 ** 9:
            return dp[i]
        res = -10 ** 9
        for j in range(1, k + 1):
            if i + j >= n:
                continue
            res = max(res, dfs(i + j))
        dp[i] = res + nums[i]
        return dp[i]

    return dfs(0)


def solution_1696_2(nums: List[int], k: int) -> int:
    """
    单调队列
    """
    n = len(nums)
    f = [0] * n
    q = deque([0])  # 双端队列
    f[0] = nums[0]
    for i in range(1, n):
        # 移出队列首元素，保证队列中所有元素都在k步内能到达i
        # 至多需要移除1位，因为i移动步长为1，每次移动都会检查是否需要移除队首
        if q[0] < i - k:
            q.popleft()
        # 维护的队列保证f[q[0]]为整个窗口中的最大值
        # 从窗口中到达i的过程中，一定是f[q[0]]到f[i]使得f[i]最大
        f[i] = f[q[0]] + nums[i]
        # 维护单调递减的性质
        while q and f[i] > f[q[-1]]:
            q.pop()
        q.append(i)
    return f[-1]
