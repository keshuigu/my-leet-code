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


def solution_9(x: int) -> bool:
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


def solution_13(s: str) -> int:
    # 将特殊情况视为2长度的字符串
    # 无字典
    # ret = 0
    # index = 0
    # while index < len(s) - 1:
    #     if s[index] == 'I':
    #         if s[index + 1] == 'V':
    #             ret += 4
    #             index += 2
    #         elif s[index + 1] == 'X':
    #             ret += 9
    #             index += 2
    #         else:
    #             ret += 1
    #             index += 1
    #     elif s[index] == 'V':
    #         ret += 5
    #         index += 1
    #     elif s[index] == 'X':
    #         if s[index + 1] == 'L':
    #             ret += 40
    #             index += 2
    #         elif s[index + 1] == 'C':
    #             ret += 90
    #             index += 2
    #         else:
    #             ret += 10
    #             index += 1
    #     elif s[index] == 'L':
    #         ret += 50
    #         index += 1
    #     elif s[index] == 'C':
    #         if s[index + 1] == 'D':
    #             ret += 400
    #             index += 2
    #         elif s[index + 1] == 'M':
    #             ret += 900
    #             index += 2
    #         else:
    #             ret += 100
    #             index += 1
    #     elif s[index] == 'D':
    #         ret += 500
    #         index += 1
    #     elif s[index] == 'M':
    #         ret += 1000
    #         index += 1
    # if index == len(s) - 1:
    #     if s[index] == 'I':
    #         ret += 1
    #     elif s[index] == 'V':
    #         ret += 5
    #     elif s[index] == 'X':
    #         ret += 10
    #     elif s[index] == 'L':
    #         ret += 50
    #     elif s[index] == 'C':
    #         ret += 100
    #     elif s[index] == 'D':
    #         ret += 500
    #     elif s[index] == 'M':
    #         ret += 1000

    # 使用字典
    f_dict = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
        'IV': 4,
        'IX': 9,
        'XL': 40,
        'XC': 90,
        'CD': 400,
        'CM': 900,
    }
    ret = 0
    index = 0
    while index < len(s) - 1:
        if s[index:index + 2] in f_dict:
            ret += f_dict[s[index:index + 2]]
            index += 2
        else:
            ret += f_dict[s[index]]
            index += 1
    if index == len(s) - 1:
        ret += f_dict[s[index]]
    return ret


def solution_13_2(s: str) -> int:
    # 注意到特殊情况仅出现在左侧数字大于右侧最大数字的情况
    # 从右往左遍历
    # 记录遇到的最大数字,遇到小数减去,遇到大数更新
    f_dict = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }
    max_num = 0
    ret = 0
    for c in s[::-1]:
        if f_dict[c] >= max_num:
            max_num = f_dict[c]
            ret += f_dict[c]
        else:
            ret -= f_dict[c]
    return ret
