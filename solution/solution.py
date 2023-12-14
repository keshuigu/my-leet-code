from typing import *


def solution_1(nums: List[int], target: int) -> List[int]:
    # 需要考虑数组中重复数组的情况
    dict_nums = {}
    for i in range(len(nums)):
        temp = target - nums[i]
        if temp in dict_nums:
            return [dict_nums[temp], i]
        dict_nums[nums[i]] = i
    return []


def solution_2(x: int) -> bool:
    if 0 <= x < 10:
        return True
    if x < 0 or x % 10 == 0:
        return False
    temp = x
    res = 0
    while res < temp:
        res = temp % 10 + res * 10
        if temp == res:
            return True
        temp = temp // 10
    if temp == res:
        return True
    else:
        return False
