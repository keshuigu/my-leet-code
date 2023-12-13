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
