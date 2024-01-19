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
    s = [1, 2, 3, 4, 5]
    s3 = [8, 9, 23, 1, 5]
    my_sort(s, s3, 0, 4)
    print(s,s3)