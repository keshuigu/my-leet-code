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
