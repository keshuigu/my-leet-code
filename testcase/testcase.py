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
