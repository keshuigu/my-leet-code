from solution import *


def testcase_1():
    print(solution_1([2, 7, 11, 15], 9))
    print(solution_1([3, 2, 4], 6))
    print(solution_1([3, 3], 6))


def testcase_9():
    print(solution_9(121))
    print(solution_9(-121))
    print(solution_9(10))


def testcase_13():
    print(solution_13('III'))
    print(solution_13('IV'))
    print(solution_13('IX'))
    print(solution_13('LVIII'))
    print(solution_13('MCMXCIV'))


def testcase_14():
    print(solution_14(["flower", "flow", "flight"]))
    print(solution_14(["dog", "racecar", "car"]))


def testcase_14_2():
    print(solution_14_2(["flower", "flow", "flight"]))
    print(solution_14_2(["dog", "racecar", "car"]))


def testcase_20():
    print(solution_20("()"))
    print(solution_20("()[]{}"))
    print(solution_20("(]"))


def testcase_21():
    print(solution_21(
        ListNode21(1, ListNode21(2, ListNode21(4))), ListNode21(1, ListNode21(3, ListNode21(4)))))
    print(solution_21(None, None))
    print(solution_21(None, ListNode21(0)))


def testcase_26():
    nums = [1, 1, 2]
    print(solution_26(nums))
    print(nums)
    nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    print(solution_26(nums))
    print(nums)


def testcase_27():
    nums = [3, 2, 2, 3]
    print(solution_27(nums, 3))
    print(nums)
    nums = [0, 1, 2, 2, 3, 0, 4, 2]
    print(solution_27(nums, 2))
    print(nums)


def testcase_28():
    print(solution_28("sadbutsad", "sad"))
    print(solution_28("leetcode", "leeto"))


def testcase_28_1():
    print(solution_28_1("sadbutsad", "sad"))
    print(solution_28_1("leetcode", "leeto"))
    print(solution_28_1("a", "a"))
    print(solution_28_1("mississippi", "issip"))


def testcase_35():
    print(solution_35([1, 3, 5, 6], 5))  # 2
    print(solution_35([1, 3, 5, 6], 2))  # 1
    print(solution_35([1, 3, 5, 6], 7))  # 4
    print(solution_35([1, 3, 5, 6], 0))  # 0
    print(solution_35([1, 3, 5, 6,8], 4))  # 2
