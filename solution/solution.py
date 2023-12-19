from typing import *
from .data_struct import *


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


def solution_14(strs: List[str]) -> str:
    # 少个找最短字符串的步骤
    ret = strs[0]
    while len(ret) > 0:
        flag = True
        for i in range(1, len(strs)):
            if len(strs[i]) < len(ret) or strs[i][:len(ret)] != ret:
                flag = False
                break
        if flag:
            return ret
        ret = ret[:-1]
    return ret


def solution_14_2(strs: List[str]) -> str:
    temp = strs[0]
    ret = temp
    for i in range(1, len(strs)):
        ret = ""
        for j in range(min(len(temp), len(strs[i]))):
            if strs[i][j] == temp[j]:
                ret += temp[j]
            else:
                break
        temp = ret
    return ret


def solution_20(s: str) -> bool:
    stack = Stack20(100000)
    # 奇数长度必为False
    if len(s) % 2 == 1:
        return False
    if len(s) == 0:
        return True
    f_dict = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    if s[0] in f_dict.keys():
        return False
    stack.push(s[0])
    index = 1
    while index < len(s):
        if s[index] in f_dict.keys():
            if not stack.is_empty() and f_dict[s[index]] == stack.peek():
                stack.pop()
            else:
                return False
        else:
            stack.push(s[index])
        index += 1
    return stack.is_empty()


def solution_21(list1: Optional[ListNode21], list2: Optional[ListNode21]) -> Optional[ListNode21]:
    if list1 is None:
        return list2
    if list2 is None:
        return list1
    if list1.val < list2.val:
        temp = ListNode21(list1.val)
        list1 = list1.next
    else:
        temp = ListNode21(list2.val)
        list2 = list2.next
    ret = temp
    while list1 is not None and list2 is not None:
        if list1.val < list2.val:
            temp.next = ListNode21(list1.val)
            temp = temp.next
            list1 = list1.next
        else:
            temp.next = ListNode21(list2.val)
            temp = temp.next
            list2 = list2.next
    while list1 is not None:
        temp.next = ListNode21(list1.val)
        temp = temp.next
        list1 = list1.next
    while list2 is not None:
        temp.next = ListNode21(list2.val)
        temp = temp.next
        list2 = list2.next
    return ret
