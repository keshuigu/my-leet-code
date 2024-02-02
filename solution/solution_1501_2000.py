import heapq
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
