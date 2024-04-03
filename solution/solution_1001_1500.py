import itertools
from typing import *
from .data_struct import *


def solution_1480(nums: List[int]) -> List[int]:
    temp = 0
    ret = list(range(len(nums)))
    for i in range(len(nums)):
        temp = temp + nums[i]
        ret[i] = temp
    return ret


def solution_1342(num: int) -> int:
    count = 0
    while num > 1:
        count = count + (num & 1) + 1
        num = num >> 1
    return count + (num & 1)


def solution_1342_2(num: int) -> int:
    if num == 0:
        return 0
    # 比较有意思的解,假定32位int
    clz = 0  # 前导0的数量,用于计算num的二进制长度
    temp = num
    if temp >> 16 == 0:
        clz += 16
        temp = temp << 16
    if temp >> 24 == 0:
        clz += 8
        temp = temp << 8
    if temp >> 28 == 0:
        clz += 4
        temp = temp << 4
    if temp >> 30 == 0:
        clz += 2
        temp = temp << 2
    if temp >> 31 == 0:
        clz += 1

    temp = num  # 求num中1的个数
    temp = (temp & 0x55555555) + ((temp >> 1) & 0x55555555)  # 计算每两位的1的个数,并保存在这两位中
    temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333)  # 将刚才的计算结果每2个一组,组成4位求和,保存在这四位中
    temp = (temp & 0x0F0F0F0F) + ((temp >> 4) & 0x0F0F0F0F)  # 同上 重复
    temp = (temp & 0x00FF00FF) + ((temp >> 8) & 0x00FF00FF)
    temp = (temp & 0x0000FFFF) + ((temp >> 16) & 0x0000FFFF)
    return 32 - clz - 1 + temp


def solution_1185(day: int, month: int, year: int) -> str:
    week = ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    day_of_year = [0, 365, 731, 1096]
    # 31, 28, 31, 30, 31, 30, 31,31, 30, 31, 30, 31
    day_of_week_1 = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    day_of_week_2 = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    # 1971.1.1 "Friday"
    temp_year = year - 1971
    temp_month = month - 1
    temp_day = day - 1
    total = (temp_year // 4) * 1461 + day_of_year[temp_year % 4]
    total += day_of_week_2[temp_month] if year % 4 == 0 else day_of_week_1[temp_month]
    total += temp_day
    if year == 2100 and month > 2:
        total -= 1
    return week[total % 7]


def solution_1185_2(day: int, month: int, year: int) -> str:
    # 蔡勒公式
    # w = (y + int(y/4) + int(c/4) -2*c +　int((13(m+1))/5) + d -1) mod 7
    week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    century = int(year / 100)
    temp_year = year - century * 100
    if month == 1 or month == 2:
        month = month + 12
        if temp_year == 0:
            temp_year = 99
            century -= 1
        else:
            temp_year = temp_year - 1
    total = temp_year + int(temp_year / 4) + int(century / 4) - 2 * century + int((26 * (month + 1)) / 10) + day - 1
    if total < 0:
        return week[(total % 7 + 7) % 7]
    else:
        return week[total % 7]


def solution_1154(date: str) -> int:
    year, month, day = date.split("-")
    year = int(year)
    month = int(month)
    day = int(day)
    day_1 = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    day_2 = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:
        return day_2[month - 1] + day
    else:
        return day_1[month - 1] + day


def solution_1094(trips: List[List[int]], capacity: int) -> bool:
    diff = [0] * 1001
    for trip in trips:
        diff[trip[1]] += trip[0]
        diff[trip[2]] -= trip[0]
    passengers = itertools.accumulate(diff)
    return max(passengers) <= capacity


def solution_1483():
    # lca
    # method.TreeAncestor
    ...


def solution_1261():
    # data_struct.FindElements
    ...


def solution_1004(nums: List[int], k: int) -> int:
    left = 0
    cnt = 0
    ans = 0
    for i in range(len(nums)):
        cnt += 1 - nums[i]
        while cnt > k:
            cnt -= 1 - nums[left]
            left += 1
        ans = max(ans, i - left + 1)
    return ans


def solution_1379(original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    def dfs(node, path):
        if node == target:
            return path
        if not node.left and not node.right:
            return None
        res1, res2 = None, None
        if node.left:
            res1 = dfs(node.left, path + '0')
        if node.right:
            res2 = dfs(node.right, path + '1')
        if not res1 and not res2:
            return None
        return res1 if res1 else res2

    p = dfs(original, "")
    cur = cloned
    for x in p:
        if x == "0":
            cur = cur.left
        else:
            cur = cur.right
    return cur


def solution_1379_2(original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
    if original is None or original is target:
        return cloned
    return solution_1379_2(original.left, cloned.left, target) or solution_1379_2(original.right, cloned.right, target)