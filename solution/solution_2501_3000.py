from typing import *
from .data_struct import *


def solution_2706(prices: List[int], money: int) -> int:
    min_1 = prices[0] if prices[0] <= prices[1] else prices[1]
    min_2 = prices[1] if prices[0] <= prices[1] else prices[0]
    for price in prices[2:]:
        if price < min_1:
            min_2 = min_1
            min_1 = price
        elif min_1 <= price < min_2:
            min_2 = price
    res = money - min_1 - min_2
    return money if res < 0 else res
