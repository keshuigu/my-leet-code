from testcase import *
import re
import sys
import os


def my_sort(nums1, nums2, left, right):
    if left >= right:
        return
    tmp = nums2[left]
    tmp1 = nums1[left]
    i = left
    j = right
    while i < j:
        while i < j and nums2[j] >= tmp:
            j -= 1
        nums2[i] = nums2[j]
        nums1[i] = nums1[j]
        while i < j and nums2[i] <= tmp:
            i += 1
        nums2[j] = nums2[i]
        nums1[j] = nums1[i]
    nums2[i] = tmp
    nums1[i] = tmp1
    my_sort(nums1, nums2, left, i - 1)
    my_sort(nums1, nums2, i + 1, right)


if __name__ == '__main__':
    p2 = [1, 3, 2]
    p1 = [[p] + [-1] * 3 for p in p2]
    print(p1)
