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
        ListNode(1, ListNode(2, ListNode(4))), ListNode(1, ListNode(3, ListNode(4)))))
    print(solution_21(None, None))
    print(solution_21(None, ListNode(0)))


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
    print(solution_35([1, 3, 5, 6, 8], 4))  # 2


def testcase_58():
    print(solution_58("Hello World"))
    print(solution_58("   fly me   to   the moon  "))
    print(solution_58("luffy is still joyboy"))


def testcase_58_2():
    print(solution_58("Hello World"))
    print(solution_58("   fly me   to   the moon  "))
    print(solution_58("luffy is still joyboy"))


def testcase_66():
    print(solution_66([1, 2, 3]))
    print(solution_66([4, 3, 2, 1]))
    print(solution_66([0]))
    print(solution_66([9, 9, 9]))
    print(solution_66([1, 9, 9]))


def testcase_67():
    print(solution_67("11", "1"))
    print(solution_67("1010", "1011"))
    print(solution_67("1111", "1111"))
    print(solution_67("1111", "1110"))
    print(solution_67("1110", "1111"))


def testcase_69():
    print(solution_69(4))
    print(solution_69(8))
    print(solution_69(9))
    print(solution_69(10))
    print(solution_69(2147395600))


def testcase_70():
    print(solution_70(2))
    print(solution_70(3))
    print(solution_70(4))  # 1.1.1.1 / 1.2.1 / 2.2 /1.1.2/2.1.1
    print(solution_70(5))
    print(solution_70(6))
    print(solution_70(7))
    print(solution_70(8))
    print(solution_70(9))
    print(solution_70(44))


def testcase_70_2():
    print(solution_70_2(2))
    print(solution_70_2(3))
    print(solution_70_2(4))  # 1.1.1.1 / 1.2.1 / 2.2 /1.1.2/2.1.1
    print(solution_70_2(5))
    print(solution_70_2(6))
    print(solution_70_2(7))
    print(solution_70_2(8))
    print(solution_70_2(9))
    print(solution_70_2(44))


def testcase_1480():
    print(solution_1480([1, 2, 3, 4]))
    print(solution_1480([1, 1, 1, 1, 1]))
    print(solution_1480([3, 1, 2, 10, 1]))


def testcase_1342():
    print(solution_1342(14))
    print(solution_1342(8))
    print(solution_1342(123))


def testcase_1342_2():
    print(solution_1342_2(14))
    print(solution_1342_2(8))
    print(solution_1342_2(123))


def testcase_1672():
    print(solution_1672([[1, 2, 3], [3, 2, 1]]))
    print(solution_1672([[1, 5], [7, 3], [3, 5]]))
    print(solution_1672([[2, 8, 7], [7, 1, 3], [1, 9, 5]]))


def testcase_2235():
    print(solution_2235(12, 5))
    print(solution_2235(-10, 4))


def testcase_412():
    print(solution_412(3))
    print(solution_412(5))
    print(solution_412(15))


def testcase_876():
    print(solution_876(
        ListNode(1,
                 ListNode(2,
                          ListNode(3,
                                   ListNode(4,
                                            ListNode(5)))))))

    print(solution_876(
        ListNode(1,
                 ListNode(2,
                          ListNode(3,
                                   ListNode(4,
                                            ListNode(5,
                                                     ListNode(6))))))))


def testcase_383():
    print(solution_383("a", "b"))
    print(solution_383("aa", "ab"))
    print(solution_383("aa", "aab"))


def testcase_2706():
    print(solution_2706([1, 2, 2], 3))
    print(solution_2706([98, 54, 6, 34, 66, 63, 52, 39], 62))
    print(solution_2706([90, 29, 6, 74], 82))


def testcase_1154():
    print(solution_1154("2019-01-09"))
    print(solution_1154("2019-02-10"))
    print(solution_1154("2000-03-11"))


def testcase_1599():
    print(solution_1599([8, 3], 5, 6))
    print(solution_1599([10, 9, 6], 6, 4))
    print(solution_1599([3, 4, 0, 5, 1], 1, 92))
    print(solution_1599([2], 2, 4))
    print(solution_1599([0, 0, 0, 0, 0, 50], 100, 1))


def testcase_83():
    print(solution_83(
        ListNode(1,
                 ListNode(1,
                          ListNode(2, None)))))

    print(solution_83(
        ListNode(1,
                 ListNode(1,
                          ListNode(2,
                                   ListNode(3,
                                            ListNode(3, None)))))))


def testcase_88():
    num1 = [1, 2, 3, 0, 0, 0]
    num2 = [2, 5, 6]
    solution_88(num1, 3, num2, 3)
    print(num1)
    num1 = [1]
    num2 = []
    solution_88(num1, 1, num2, 0)
    print(num1)
    num1 = [0]
    num2 = [1]
    solution_88(num1, 0, num2, 1)
    print(num1)


def testcase_94():
    print(solution_94(TreeNode(1, None,
                               TreeNode(2,
                                        TreeNode(3, None, None), None))))
    print(solution_94(TreeNode(1, TreeNode(4, None, None),
                               TreeNode(2,
                                        TreeNode(3, None, TreeNode(5, None, None)), None))))


def testcase_94_2():
    print(solution_94_2(TreeNode(1, None,
                                 TreeNode(2,
                                          TreeNode(3, None, None), None))))
    print(solution_94_2(TreeNode(1, TreeNode(4, None, None),
                                 TreeNode(2,
                                          TreeNode(3, None, TreeNode(5, None, None)), None))))


def testcase_466():
    print(solution_466("acb", 4, "ab", 2))
    print(solution_466("acb", 1, "acb", 1))
    print(solution_466("abdbec", 1, "abc", 1))
    print(solution_466("asc", 3, "ca", 1))
    print(solution_466("ecbafedcba", 4, "abcdef", 1))
    print(solution_466("niconiconi", 99981, "nico", 81))
    print(solution_466(
        "phqghumeaylnlfdxfircvscxggbwkfnqduxwfnfozvsrtkjprepggxrpnrvystmwcysyycqpevikeffmznimkkasvwsrenzkycxf",
        1000000,
        "xtlsgypsfadpooefxzbcoejuvpvaboygpoeylfpbnpljvrvipyamyehwqnqrqpmxujjloovaowuxwhmsncbxcoksfzkvatxdknly",
        100))


def testcase_466_2():
    print(solution_466_2("acb", 4, "ab", 2))
    print(solution_466_2("acb", 1, "acb", 1))
    print(solution_466_2("abdbec", 1, "abc", 1))
    print(solution_466_2("asc", 3, "ca", 1))
    print(solution_466_2("ecbafedcba", 4, "abcdef", 1))
    print(solution_466_2("niconiconi", 99981, "nico", 81))
    print(solution_466_2(
        "phqghumeaylnlfdxfircvscxggbwkfnqduxwfnfozvsrtkjprepggxrpnrvystmwcysyycqpevikeffmznimkkasvwsrenzkycxf",
        1000000,
        "xtlsgypsfadpooefxzbcoejuvpvaboygpoeylfpbnpljvrvipyamyehwqnqrqpmxujjloovaowuxwhmsncbxcoksfzkvatxdknly",
        100))


def testcase_100():
    print(solution_100(TreeNode(1, None,
                                TreeNode(2,
                                         TreeNode(3, None, None), None)),
                       TreeNode(1, None,
                                TreeNode(2,
                                         TreeNode(3, None, None), None))
                       ))
    print(solution_100(TreeNode(1, TreeNode(4, None, None),
                                TreeNode(2,
                                         TreeNode(3, None, TreeNode(5, None, None)), None)),
                       TreeNode(1, TreeNode(4, None, None),
                                TreeNode(2,
                                         TreeNode(3, None, TreeNode(5, None, None)), None))))


def testcase_101():
    print(solution_101(TreeNode(1, TreeNode(2, TreeNode(3), TreeNode(4)),
                                TreeNode(2,
                                         TreeNode(4), TreeNode(3)))
                       ))
    print(solution_101(TreeNode(1,
                                TreeNode(2, None, TreeNode(3)),
                                TreeNode(2, None, TreeNode(3)))
                       ))
    # [9,-42,-42,null,76,76,null,null,13,null,13]
    print(solution_101(TreeNode(9,
                                TreeNode(-42, None, TreeNode(76, None, TreeNode(13))),
                                TreeNode(-42, TreeNode(76, None, TreeNode(13)), None))
                       ))


def testcase_2487():
    print(solution_2487(
        ListNode(5, ListNode(2, ListNode(3, ListNode(13, ListNode(3, ListNode(8))))))
    ))
    print(solution_2487(
        ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1))))))
    ))


def testcase_2487_2():
    print(solution_2487_2(
        ListNode(5, ListNode(2, ListNode(3, ListNode(13, ListNode(3, ListNode(8))))))
    ))
    print(solution_2487_2(
        ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1))))))
    ))


def testcase_2487_3():
    print(solution_2487_3(
        ListNode(5, ListNode(2, ListNode(3, ListNode(13, ListNode(3, ListNode(8))))))
    ))
    print(solution_2487_3(
        ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1))))))
    ))


def testcase_104():
    print(solution_104(
        TreeNode(3, TreeNode(9),
                 TreeNode(20, TreeNode(15), TreeNode(7))
                 )))
    print(solution_104(
        TreeNode(1, None, TreeNode(2))
    ))


def testcase_2397():
    print(solution_2397([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]], 2))
    print(solution_2397([[1], [0]], 1))


def testcase_2397_2():
    print(solution_2397_2([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]], 2))
    print(solution_2397_2([[1], [0]], 1))


def testcase_108():
    print(solution_108([-10, -3, 0, 5, 9]))
    print(solution_108([1, 3]))


def testcase_1944():
    print(solution_1944([10, 6, 8, 5, 11, 9]))
    print(solution_1944([5, 1, 2, 3, 10]))


def testcase_110():
    print(solution_110(TreeNode(3, TreeNode(9),
                                TreeNode(20, TreeNode(15), TreeNode(7))
                                )))
    print(solution_110(TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4)), TreeNode(3)),
                                TreeNode(2)
                                )))


def testcase_111():
    print(solution_111(TreeNode(3, TreeNode(9),
                                TreeNode(20, TreeNode(15), TreeNode(7))
                                )))
    print(solution_111(TreeNode(2, None, TreeNode(3, None, TreeNode(4, None, TreeNode(5, None, TreeNode(6)))))))


def testcase_112():
    print(solution_112(TreeNode(3, TreeNode(9),
                                TreeNode(20, TreeNode(15), TreeNode(7))
                                ), 12))
    print(solution_112(TreeNode(2, None, TreeNode(3, None, TreeNode(4, None, TreeNode(5, None, TreeNode(6))))), 20))
    print(solution_112(TreeNode(1, TreeNode(2), TreeNode(3)), 5))
    print(solution_112(TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))),
                                TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1)))), 22))


def testcase_2807():
    print(solution_2807(ListNode(5, ListNode(2, ListNode(6, ListNode(18, ListNode(3, ListNode(8))))))))
    print(solution_2807(ListNode(18, ListNode(6, ListNode(10, ListNode(3))))))


def testcase_118():
    print(solution_118(5))
    print(solution_118(1))


def testcase_119():
    print(solution_119(3))
    print(solution_119(0))
    print(solution_119(33))


def testcase_121():
    print(solution_121([7, 1, 5, 3, 6, 4]))
    print(solution_121([7, 6, 4, 3, 1]))


def testcase_125():
    print(solution_125("A man, a plan, a canal: Panama"))
    print(solution_125("race a car"))
    print(solution_125(" "))
    print(solution_125("0P"))


def testcase_136():
    print(solution_136([2, 2, 1]))
    print(solution_136([4, 1, 2, 1, 2]))


def testcase_447():
    print(solution_447([[0, 0], [1, 0], [2, 0], [-1, 0]]))
    print(solution_447([[1, 1], [2, 2], [3, 3], [0, 0]]))
    print(solution_447([[1, 1]]))


def testcase_141():
    tail = ListNode(-4)
    mid = ListNode(2, ListNode(0, tail))
    head = ListNode(3, mid)
    tail.next = mid
    print(solution_141(head))
    tail = ListNode(2)
    mid = ListNode(1, tail)
    head = mid
    tail.next = mid
    print(solution_141(head))
    head = ListNode(1)
    print(solution_141(head))


def testcase_144():
    print(solution_144(TreeNode(3, TreeNode(9),
                                TreeNode(20, TreeNode(15), TreeNode(7))
                                )))
    print(solution_144(TreeNode(2, None, TreeNode(3, None, TreeNode(4, None, TreeNode(5, None, TreeNode(6)))))))
    print(solution_144(TreeNode(1, TreeNode(2), TreeNode(3))))
    print(solution_144(TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))),
                                TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1))))))


def testcase_144_2():
    print(solution_144_2(TreeNode(3, TreeNode(9),
                                  TreeNode(20, TreeNode(15), TreeNode(7))
                                  )))
    print(solution_144_2(TreeNode(2, None, TreeNode(3, None, TreeNode(4, None, TreeNode(5, None, TreeNode(6)))))))
    print(solution_144_2(TreeNode(1, TreeNode(2), TreeNode(3))))
    print(solution_144_2(TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))),
                                  TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1))))))


def testcase_145():
    print(solution_145(TreeNode(3, TreeNode(9),
                                TreeNode(20, TreeNode(15), TreeNode(7))
                                )))
    print(solution_145(TreeNode(2, None, TreeNode(3, None, TreeNode(4, None, TreeNode(5, None, TreeNode(6)))))))
    print(solution_145(TreeNode(1, TreeNode(2), TreeNode(3))))
    print(solution_145(TreeNode(5, TreeNode(4, TreeNode(11, TreeNode(7), TreeNode(2))),
                                TreeNode(8, TreeNode(13), TreeNode(4, None, TreeNode(1))))))


def testcase_208():
    trie = Trie()
    trie.insert("apple")
    print(trie.search("apple"))
    print(trie.search("app"))
    print(trie.starts_with("app"))
    trie.insert("app")
    print(trie.search("app"))


def testcase_2707():
    print(solution_2707("leetscode", ["leet", "code", "leetcode"]))
    print(solution_2707("sayhelloworld", ["hello", "world"]))


def testcase_160():
    headA = ListNode(4, ListNode(1))
    headB = ListNode(5, ListNode(6, ListNode(1)))
    mid = ListNode(8, ListNode(4, ListNode(5)))
    headA.next = mid
    headB.next = mid
    print(solution_160(headA, headB))
    headA = ListNode(1, ListNode(9, ListNode(1)))
    headB = ListNode(3)
    mid = ListNode(2, ListNode(4))
    headA.next = mid
    headB.next = mid
    print(solution_160(headA, headB))
    headA = ListNode(2, ListNode(6, ListNode(4)))
    headB = ListNode(1, ListNode(5))
    print(solution_160(headA, headB))


def testcase_160_2():
    headA = ListNode(4, ListNode(1))
    headB = ListNode(5, ListNode(6, ListNode(1)))
    mid = ListNode(8, ListNode(4, ListNode(5)))
    headA.next = mid
    headB.next = mid
    print(solution_160_2(headA, headB))
    headA = ListNode(1, ListNode(9, ListNode(1)))
    headB = ListNode(3)
    mid = ListNode(2, ListNode(4))
    headA.next = mid
    headB.next = mid
    print(solution_160_2(headA, headB))
    headA = ListNode(2, ListNode(6, ListNode(4)))
    headB = ListNode(1, ListNode(5))
    print(solution_160_2(headA, headB))


def testcase_168():
    print(solution_168(1))
    print(solution_168(28))
    print(solution_168(70))
    print(solution_168(2147483647))


def testcase_171():
    print(solution_171("A"))
    print(solution_171("AB"))
    print(solution_171("ZY"))
    print(solution_171("FXSHRXW"))
