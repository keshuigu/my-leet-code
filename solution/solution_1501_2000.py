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
