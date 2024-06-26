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


def testcase_2696():
    print(solution_2696("ABFCACDB"))
    print(solution_2696("ACBBD"))
    print(solution_2696("CDABCDD"))


def testcase_169():
    print(solution_169([3, 2, 3]))
    print(solution_169([2, 2, 1, 1, 1, 2, 2]))


def testcase_175():
    preprocess(
        """DROP TABLE Person;
        DROP TABLE Address;
        Create table If Not Exists Person (personId int, firstName varchar(255), lastName varchar(255));
        Create table If Not Exists Address (addressId int, personId int, city varchar(255), state varchar(255));
        Truncate table Person;
        insert into Person (personId, lastName, firstName) values ('1', 'Wang', 'Allen');
        insert into Person (personId, lastName, firstName) values ('2', 'Alice', 'Bob');
        Truncate table Address;
        insert into Address (addressId, personId, city, state) values ('1', '2', 'New York City', 'New York');
        insert into Address (addressId, personId, city, state) values ('2', '3', 'Leetcode', 'California');""")
    print(solution_175())


def testcase_181():
    preprocess(
        """DROP TABLE Employee;
        Create table If Not Exists Employee (id int, name varchar(255), salary int, managerId int);
        Truncate table Employee;
        insert into Employee (id, name, salary, managerId) values ('1', 'Joe', '70000', '3');
        insert into Employee (id, name, salary, managerId) values ('2', 'Henry', '80000', '4');
        insert into Employee (id, name, salary, managerId) values ('3', 'Sam', '60000', null);
        insert into Employee (id, name, salary, managerId) values ('4', 'Max', '90000', null);""")
    print(solution_181())


def testcase_182():
    preprocess(
        """DROP TABLE Person;
        Create table If Not Exists Person (id int, email varchar(255));
        Truncate table Person;
        insert into Person (id, email) values ('1', 'a@b.com');
        insert into Person (id, email) values ('2', 'c@d.com');
        insert into Person (id, email) values ('3', 'a@b.com');""")
    print(solution_182())


def testcase_183():
    preprocess(
        """Drop table Customers;
        DROP table Orders;
        Create table If Not Exists Customers (id int, name varchar(255));
        Create table If Not Exists Orders (id int, customerId int);
        Truncate table Customers;
        insert into Customers (id, name) values ('1', 'Joe');
        insert into Customers (id, name) values ('2', 'Henry');
        insert into Customers (id, name) values ('3', 'Sam');
        insert into Customers (id, name) values ('4', 'Max');
        Truncate table Orders;
        insert into Orders (id, customerId) values ('1', '3');
        insert into Orders (id, customerId) values ('2', '1');""")
    print(solution_183())


def testcase_2645():
    print(solution_2645("b"))
    print(solution_2645("aaa"))
    print(solution_2645("abc"))


def testcase_2645_2():
    print(solution_2645_2("b"))
    print(solution_2645_2("aaa"))
    print(solution_2645_2("abc"))


def testcase_2645_3():
    print(solution_2645_3("b"))
    print(solution_2645_3("aaa"))
    print(solution_2645_3("abc"))


def testcase_2645_4():
    print(solution_2645_4("b"))
    print(solution_2645_4("aaa"))
    print(solution_2645_4("abc"))


def testcase_190():
    print(solution_190(int("11111111111111111111111111111101", 2)))
    print(solution_190(int("00000010100101000001111010011100", 2)))


def testcase_190_2():
    print(solution_190_2(int("11111111111111111111111111111101", 2)))
    print(solution_190_2(int("00000010100101000001111010011100", 2)))


def testcase_191():
    print(solution_191(int("11111111111111111111111111111101", 2)))
    print(solution_191(int("01011", 2)))
    print(solution_191(int("1000000", 2)))


def testcase_193():
    with open("resources/file.txt", "w") as f:
        f.write("987-123-4567\n123 456 7890\n(123) 456-7890\n0(001) 345-0000\n")
    solution_193()


def testcase_195():
    with open("resources/file.txt", "w") as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6\nLine 7\nLine 8\nLine 9\nLine 10\n")
    solution_195()


def testcase_2085():
    print(solution_2085(["leetcode", "is", "amazing", "as", "is"], ["amazing", "leetcode", "is"]))
    print(solution_2085(["b", "bb", "bbb"], ["a", "aa", "aaa"]))
    print(solution_2085(["a", "ab"], ["a", "a", "a", "ab"]))


def testcase_202():
    print(solution_202(19))
    print(solution_202(2))


def testcase_202_2():
    print(solution_202_2(19))
    print(solution_202_2(2))


def testcase_203():
    print(solution_203(ListNode(1, ListNode(2, ListNode(6, ListNode(3, ListNode(4, ListNode(5, ListNode(6))))))), 6))
    print(solution_203(None, 1))
    print(solution_203(ListNode(7, ListNode(7, ListNode(7, ListNode(7, ListNode(7))))), 7))


def testcase_2182():
    print(solution_2182("cczazcc", 3))
    print(solution_2182("aababab", 2))
    print(solution_2182("bbbcccaaaad", 3))


def testcase_2182_2():
    print(solution_2182_2("cczazcc", 3))
    print(solution_2182_2("aababab", 2))
    print(solution_2182_2("bbbcccaaaad", 3))


def testcase_205():
    print(solution_205("egg", "add"))
    print(solution_205("foo", "bar"))
    print(solution_205("paper", "title"))
    print(solution_205("badc", "baba"))


def testcase_290():
    print(solution_290("abba", "dog cat cat dog"))
    print(solution_290("abba", "dog cat cat fish"))
    print(solution_290("aaaa", "dog cat cat dog"))


def testcase_2():
    print(solution_2(ListNode(2, ListNode(4, ListNode(3))),
                     ListNode(5, ListNode(6, ListNode(4)))))
    print(solution_2(ListNode(0),
                     ListNode(0)))
    print(solution_2(
        ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9, ListNode(9)))))))),
        ListNode(9, ListNode(9, ListNode(9, ListNode(9))))))


def testcase_3():
    print(solution_3("abcabcbb"))
    print(solution_3("bbbbb"))
    print(solution_3("pwwkew"))
    print(solution_3("abba"))
    print(solution_3("abdsdcestgad"))


def testcase_3_2():
    print(solution_3_2("abcabcbb"))
    print(solution_3_2("bbbbb"))
    print(solution_3_2("pwwkew"))
    print(solution_3_2("abba"))
    print(solution_3_2("abdsdcestgad"))


def testcase_3_3():
    print(solution_3_3("abcabcbb"))
    print(solution_3_3("bbbbb"))
    print(solution_3_3("pwwkew"))
    print(solution_3_3("abba"))
    print(solution_3_3("abdsdcestgad"))
    print(solution_3_3(" "))


def testcase_82():
    print(solution_82(ListNode(1, ListNode(2, ListNode(3, ListNode(3, ListNode(4, ListNode(4, ListNode(5)))))))))
    print(solution_82(ListNode(1, ListNode(1, ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))))
    print(solution_82(ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(1, ListNode(5)))))))))
    print(solution_82(ListNode(1, ListNode(1, ListNode(2, ListNode(2, ListNode(3, ListNode(4, ListNode(4)))))))))


def testcase_196():
    preprocess(
        """Drop table Person;
        Create table If Not Exists Person (Id int, Email varchar(255));
        Truncate table Person;
        insert into Person (id, email) values ('1', 'john@example.com');
        insert into Person (id, email) values ('2', 'bob@example.com');
        insert into Person (id, email) values ('3', 'john@example.com')""")
    print(solution_196())
    print(execute("select * from Person"))


def testcase_197():
    preprocess(
        """Drop table Weather;
        Create table If Not Exists Weather (id int, recordDate date, temperature int);
        Truncate table Weather;
        insert into Weather (id, recordDate, temperature) values ('1', '2015-01-01', '10');
        insert into Weather (id, recordDate, temperature) values ('2', '2015-01-02', '25');
        insert into Weather (id, recordDate, temperature) values ('3', '2015-01-03', '20');
        insert into Weather (id, recordDate, temperature) values ('4', '2015-01-04', '30');""")
    print(solution_197())


def testcase_206():
    print(solution_206(ListNode(1, ListNode(2, ListNode(3, ListNode(3, ListNode(4, ListNode(4, ListNode(5)))))))))
    print(solution_206(ListNode(1, ListNode(2))))
    print(solution_206(None))


def testcase_217():
    print(solution_217([1, 2, 3, 1]))
    print(solution_217([1, 2, 3, 4]))
    print(solution_217([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))


def testcase_2719():
    print(solution_2719("1", "12", 1, 8))
    print(solution_2719("3", "5", 1, 5))
    print(solution_2719("1", "12345688", 1, 300))


def testcase_2744():
    print(solution_2744(["cd", "ac", "dc", "ca", "zz"]))
    print(solution_2744(["ab", "ba", "cc"]))
    print(solution_2744(["aa", "ab"]))
    print(solution_2744(["ff", "tx", "qr", "zw", "wr", "jr", "zt", "jk", "sq", "xx"]))


def testcase_2744_2():
    print(solution_2744_2(["cd", "ac", "dc", "ca", "zz"]))
    print(solution_2744_2(["ab", "ba", "cc"]))
    print(solution_2744_2(["aa", "ab"]))
    print(solution_2744_2(["ff", "tx", "qr", "zw", "wr", "jr", "zt", "jk", "sq", "xx"]))


def testcase_2376():
    # print(solution_2376(20))
    # print(solution_2376(5))
    # print(solution_2376(135))
    print(solution_2376(233))


def testcase_2376_2():
    print(solution_2376_2(20))
    print(solution_2376_2(5))
    print(solution_2376_2(135))
    print(solution_2376_2(233))


def testcase_233():
    print(solution_233(13))
    print(solution_233(0))


def testcase_2171():
    print(solution_2171([4, 1, 6, 5]))
    print(solution_2171([2, 10, 3, 2]))


def testcase_iq_17_06():
    print(solution_iq_17_06(25))


def testcase_600():
    print(solution_600(5))
    print(solution_600(1))
    print(solution_600(2))
    print(solution_600(4))


def testcase_219():
    print(solution_219([1, 2, 3, 1], 3))
    print(solution_219([1, 0, 1, 1], 1))
    print(solution_219([1, 2, 3, 1, 2, 3], 2))


def testcase_219_2():
    print(solution_219_2([1, 2, 3, 1], 3))
    print(solution_219_2([1, 0, 1, 1], 1))
    print(solution_219_2([1, 2, 3, 1, 2, 3], 2))


def testcase_2809():
    print(solution_2809([1, 2, 3], [1, 2, 3], 4))
    print(solution_2809([1, 2, 3], [3, 3, 3], 4))
    print(solution_2809([7, 9, 8, 5, 8, 3], [0, 1, 4, 2, 3, 1], 37))


def testcase_222():
    print(solution_222(TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, TreeNode(6)))))
    print(solution_222(None))
    print(solution_222(TreeNode(1)))


def testcase_222_2():
    print(solution_222_2(TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, TreeNode(6)))))
    print(solution_222_2(None))
    print(solution_222_2(TreeNode(1)))


def testcase_225():
    s = StackWithQueue()
    s.push(1)
    s.push(2)
    print(s.top())
    print(s.pop())
    print(s.empty())


def testcase_902():
    print(solution_902(digits=["1", "3", "5", "7"], n=100))
    print(solution_902(digits=["1", "4", "9"], n=1000000000))


def testcase_2788():
    print(solution_2788(words=["one.two.three", "four.five", "six"], separator="."))
    print(solution_2788(words=["$easy$", "$problem$"], separator="$"))
    print(solution_2788(words=["|||"], separator="|"))


def testcase_2788_2():
    print(solution_2788_2(words=["one.two.three", "four.five", "six"], separator="."))
    print(solution_2788_2(words=["$easy$", "$problem$"], separator="$"))
    print(solution_2788_2(words=["|||"], separator="|"))


def testcase_226():
    print(solution_226(TreeNode(2, TreeNode(3), TreeNode(1))))
    print(solution_226(TreeNode(4, TreeNode(2, TreeNode(1), TreeNode(3)), TreeNode(7, TreeNode(6), TreeNode(9)))))


def testcase_228():
    print(solution_228([0, 1, 2, 4, 5, 7]))
    print(solution_228([0, 2, 3, 4, 6, 8, 9]))
    print(solution_228([]))


def testcase_410():
    print(solution_410([7, 2, 5, 10, 8], 2))
    print(solution_410([1, 2, 3, 4, 5], 2))
    print(solution_410([1, 4, 4], 3))
    print(solution_410([1, 2, 3, 4, 5, 6], 4))


def testcase_410_2():
    print(solution_410_2([7, 2, 5, 10, 8], 2))
    print(solution_410_2([1, 2, 3, 4, 5], 2))
    print(solution_410_2([1, 4, 4], 3))
    print(solution_410_2([1, 2, 3, 4, 5, 6], 4))


def testcase_410_3():
    print(solution_410_3([7, 2, 5, 10, 8], 2))
    print(solution_410_3([1, 2, 3, 4, 5], 2))
    print(solution_410_3([1, 4, 4], 3))
    print(solution_410_3([1, 2, 3, 4, 5, 6], 4))


def testcase_3014():
    print(solution_3014("abcde"))
    print(solution_3014("xycdefghij"))


def testcase_3016():
    print(solution_3016("abcde"))
    print(solution_3016("xyzxyzxyzxyz"))
    print(solution_3016("aabbccddeeffgghhiiiiii"))


def testcase_3016_2():
    print(solution_3016_2("abcde"))
    print(solution_3016_2("xyzxyzxyzxyz"))
    print(solution_3016_2("aabbccddeeffgghhiiiiii"))


def testcase_3015():
    print(solution_3015(3, 1, 3))
    print(solution_3015(5, 2, 4))
    print(solution_3015(4, 1, 1))
    print(solution_3015(50, 12, 4))
    print(solution_3015(5, 1, 5))


def testcase_3015_2():
    print(solution_3015_2(3, 1, 3))
    print(solution_3015_2(5, 2, 4))
    print(solution_3015_2(4, 1, 1))
    print(solution_3015_2(50, 12, 4))
    print(solution_3015_2(5, 1, 5))


def testcase_3017():
    print(solution_3017(3, 1, 3))
    print(solution_3017(5, 2, 4))
    print(solution_3017(4, 1, 1))
    print(solution_3017(50, 12, 4))
    print(solution_3017(5, 1, 5))


def testcase_3017_2():
    print(solution_3017_2(3, 1, 3))
    print(solution_3017_2(5, 2, 4))
    print(solution_3017_2(4, 1, 1))
    print(solution_3017_2(50, 12, 4))
    print(solution_3017_2(5, 1, 5))


def testcase_670():
    print(solution_670(2736))
    print(solution_670(9973))
    print(solution_670(1993))
    print(solution_670(99901))
    print(solution_670(99910))
    print(solution_670(99911))
    print(solution_670(99919))
    print(solution_670(99999))


def testcase_670_2():
    print(solution_670_2(2736))
    print(solution_670_2(9973))
    print(solution_670_2(1993))
    print(solution_670_2(99901))
    print(solution_670_2(99910))
    print(solution_670_2(99911))
    print(solution_670_2(99919))
    print(solution_670_2(99999))


def testcase_2765():
    print(solution_2765([2, 3, 4, 3, 4]))
    print(solution_2765([2, 3, 5, 3, 4, 3, 2, 6, 7, 6, 7, 6, 7]))
    print(solution_2765([4, 5, 6]))
    print(solution_2765([21, 7, 9]))
    print(solution_2765([14, 30, 29, 49, 3, 23, 44, 21, 26, 52]))
    print(solution_2765([6, 12, 2, 3, 8, 9, 10, 10, 2, 1]))


def testcase_1094():
    print(solution_1094(trips=[[2, 1, 5], [3, 3, 7]], capacity=4))
    print(solution_1094(trips=[[2, 1, 5], [3, 3, 7]], capacity=5))
    print(solution_1094(
        trips=[[1, 1, 1000], [1, 1, 1000], [1, 1, 1000], [1, 1, 1000], [1, 1, 1000], [1, 1, 1000], [1, 1, 1000],
               [1, 1, 1000], [1, 1, 1000], [1, 1, 1000]], capacity=5))


def testcase_231():
    print(solution_231(1))
    print(solution_231(16))
    print(solution_231(3))
    print(solution_231(4))
    print(solution_231(-4))


def testcase_231_2():
    print(solution_231_2(1))
    print(solution_231_2(16))
    print(solution_231_2(3))
    print(solution_231_2(4))
    print(solution_231_2(-4))


def testcase_232():
    qws = QueueWithStack()
    qws.push(1)
    qws.push(2)
    print(qws.peek())
    print(qws.pop())
    print(qws.empty())
    print(qws.pop())
    print(qws.empty())


def testcase_232_2():
    qws = QueueWithStack2()
    qws.push(1)
    qws.push(2)
    print(qws.peek())
    print(qws.pop())
    print(qws.empty())
    print(qws.pop())
    print(qws.empty())


def testcase_2865():
    print(solution_2865([5, 3, 4, 1, 1]))
    print(solution_2865([6, 5, 3, 9, 2, 7]))
    print(solution_2865([3, 2, 5, 5, 2, 3]))
    print(solution_2865([3, 3, 3, 3, 1, 5, 1, 2, 2]))


def testcase_2865_2():
    print(solution_2865_2([5, 3, 4, 1, 1]))
    print(solution_2865_2([6, 5, 3, 9, 2, 7]))
    print(solution_2865_2([3, 2, 5, 5, 2, 3]))
    print(solution_2865_2([3, 3, 3, 3, 1, 5, 1, 2, 2]))


def testcase_234():
    print(solution_234(ListNode(2, ListNode(3, ListNode(3, ListNode(2))))))
    print(solution_234(ListNode(1, ListNode(2, ListNode(3, ListNode(2, ListNode(1)))))))
    print(solution_234(ListNode(1, ListNode(2))))


def testcase_242():
    print(solution_242("anagram", "nagaram"))
    print(solution_242("car", "rat"))


def testcase_2859():
    print(solution_2859([5, 10, 1, 5, 2], 1))
    print(solution_2859([4, 3, 2, 1], 2))
    print(solution_2859([7, 3], 0))
    print(solution_2859([1, 2, 4, 1, 9, 7, 7, 6], 3))


def testcase_2859_2():
    print(solution_2859_2([5, 10, 1, 5, 2], 1))
    print(solution_2859_2([4, 3, 2, 1], 2))
    print(solution_2859_2([7, 3], 0))
    print(solution_2859_2([1, 2, 4, 1, 9, 7, 7, 6], 3))


def testcase_2859_3():
    print(solution_2859_3([5, 10, 1, 5, 2], 1))
    print(solution_2859_3([4, 3, 2, 1], 2))
    print(solution_2859_3([7, 3], 0))
    print(solution_2859_3([1, 2, 4, 1, 9, 7, 7, 6], 3))


def testcase_lca():
    print(lca_simple([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 7, 4))
    print(lca_simple([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 5, 1))
    print(lca_simple([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 2, 6))
    print(lca_simple([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 7, 6))
    print(lca_simple([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 5, 3))


def testcase_lca_2():
    print(lca_bin_lift([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 7, 4))
    print(lca_bin_lift([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 5, 1))
    print(lca_bin_lift([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 2, 6))
    print(lca_bin_lift([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 7, 6))
    print(lca_bin_lift([[0, 1], [0, 2], [7, 1], [3, 1], [4, 1], [3, 5], [4, 6]], 5, 3))


def testcase_1483():
    tree = TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2])
    print(tree.getKthAncestor(3, 1))
    print(tree.getKthAncestor(5, 2))
    print(tree.getKthAncestor(6, 3))


def testcase_2846():
    print(solution_2846(n=7, edges=[[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 2], [4, 5, 2], [5, 6, 2]],
                        queries=[[0, 3], [3, 6], [2, 6], [0, 6]]))
    print(solution_2846(n=8, edges=[[1, 2, 6], [1, 3, 4], [2, 4, 6], [2, 5, 3], [3, 6, 6], [3, 0, 8], [7, 0, 2]],
                        queries=[[4, 6], [0, 4], [6, 5], [7, 4]]))


def testcase_2861():
    print(solution_2861(n=3, k=2, budget=15, composition=[[1, 1, 1], [1, 1, 10]], stock=[0, 0, 0], cost=[1, 2, 3]))
    print(solution_2861(n=3, k=2, budget=15, composition=[[1, 1, 1], [1, 1, 10]], stock=[0, 0, 100], cost=[1, 2, 3]))
    print(solution_2861(n=2, k=3, budget=10, composition=[[2, 1], [1, 2], [1, 1]], stock=[1, 1], cost=[5, 5]))


def testcase_275():
    print(solution_275([0, 1, 3, 5, 6]))
    print(solution_275([1, 2, 100]))


def testcase_365():
    print(solution_365(3, 5, 4))
    print(solution_365(2, 6, 5))
    print(solution_365(1, 2, 3))


def testcase_365_2():
    print(solution_365_2(3, 5, 4))
    print(solution_365_2(2, 6, 5))
    print(solution_365_2(1, 2, 3))


def testcase_3019():
    print(solution_3019("aAbBcC"))
    print(solution_3019("AaAaAaaA"))


def testcase_3020():
    print(solution_3020([5, 4, 1, 2, 2]))
    print(solution_3020([1, 3, 2, 4]))
    print(solution_3020([1, 3, 9, 81, 81, 9, 3, 4]))
    print(solution_3020([1, 3, 9, 81, 3, 4]))
    print(solution_3020([1, 1]))


def testcase_3020_2():
    print(solution_3020_2([5, 4, 1, 2, 2]))
    print(solution_3020_2([1, 3, 2, 4]))
    print(solution_3020_2([1, 3, 9, 81, 81, 9, 3, 4]))
    print(solution_3020_2([1, 3, 9, 81, 3, 4]))
    print(solution_3020_2([1, 1]))


def testcase_3021():
    print(solution_3021(3, 5))


def testcase_3022():
    print(solution_3022(nums=[3, 5, 3, 2, 7], k=2))
    print(solution_3022(nums=[7, 3, 15, 14, 2, 8], k=4))
    print(solution_3022(nums=[10, 7, 10, 3, 9, 14, 9, 4], k=1))


def testcase_514():
    print(solution_514("godding", "gd"))
    print(solution_514("godding", "godding"))


def testcase_514_2():
    print(solution_514_2("godding", "gd"))
    print(solution_514_2("godding", "godding"))


def testcase_2808():
    print(solution_2808([1, 2, 1, 2]))
    print(solution_2808([2, 1, 3, 3, 2]))
    print(solution_2808([5, 5, 5, 5]))
    print(solution_2808([15, 14, 14, 19]))


def testcase_2808_2():
    print(solution_2808_2([1, 2, 1, 2]))
    print(solution_2808_2([2, 1, 3, 3, 2]))
    print(solution_2808_2([5, 5, 5, 5]))
    print(solution_2808_2([15, 14, 14, 19]))


def testcase_2808_3():
    print(solution_2808_3([1, 2, 1, 2]))
    print(solution_2808_3([2, 1, 3, 3, 2]))
    print(solution_2808_3([5, 5, 5, 5]))
    print(solution_2808_3([15, 14, 14, 19]))


def testcase_2670():
    print(solution_2670([1, 2, 3, 4, 5]))
    print(solution_2670([1, 2, 1, 4, 5]))


def testcase_lcp_24():
    print(solution_lcp_24([3, 4, 5, 1, 6, 7]))
    print(solution_lcp_24([1, 2, 3, 4, 5]))
    print(solution_lcp_24([1, 1, 1, 2, 3, 4]))


def testcase_1686():
    print(solution_1686(aliceValues=[1, 3], bobValues=[2, 1]))
    print(solution_1686(aliceValues=[1, 2], bobValues=[3, 1]))
    print(solution_1686(aliceValues=[2, 4, 3], bobValues=[1, 6, 7]))


def testcase_1686_2():
    print(solution_1686_2(aliceValues=[1, 3], bobValues=[2, 1]))
    print(solution_1686_2(aliceValues=[1, 2], bobValues=[3, 1]))
    print(solution_1686_2(aliceValues=[2, 4, 3], bobValues=[1, 6, 7]))


def testcase_1690():
    print(solution_1690([5, 3, 1, 4, 2]))
    print(solution_1690([7, 90, 5, 1, 100, 10, 10, 2]))


def testcase_1690_2():
    print(solution_1690_2([5, 3, 1, 4, 2]))
    print(solution_1690_2([7, 90, 5, 1, 100, 10, 10, 2]))


def testcase_3024():
    print(solution_3024([3, 3, 3]))
    print(solution_3024([3, 4, 5]))


def testcase_3024_2():
    print(solution_3024_2([3, 3, 3]))
    print(solution_3024_2([3, 4, 5]))


def testcase_3025():
    print(solution_3025([[1, 1], [2, 2], [3, 3]]))
    print(solution_3025([[6, 2], [4, 4], [2, 6]]))
    print(solution_3025([[3, 1], [1, 3], [1, 1]]))


def testcase_3026():
    print(solution_3026(nums=[1, 2, 3, 4, 5, 6], k=1))
    print(solution_3026(nums=[-1, 3, 2, 4, 5], k=3))
    print(solution_3026(nums=[-1, -2, -3, -4], k=2))


def testcase_3026_2():
    print(solution_3026_2(nums=[1, 2, 3, 4, 5, 6], k=1))
    print(solution_3026_2(nums=[-1, 3, 2, 4, 5], k=3))
    print(solution_3026_2(nums=[-1, -2, -3, -4], k=2))


def testcase_292():
    print(solution_292(4))
    print(solution_292(1))
    print(solution_292(2))
    print(solution_292(5))


def testcase_292_2():
    print(solution_292_2(4))
    print(solution_292_2(1))
    print(solution_292_2(2))
    print(solution_292_2(5))


def testcase_3028():
    print(solution_3028([2, 3, -5]))
    print(solution_3028([3, 2, 3, -4]))


def testcase_3029():
    print(solution_3029(word="abacaba", k=3))
    print(solution_3029(word="abacaba", k=4))
    print(solution_3029(word="abcbabcd", k=2))


def testcase_3031():
    print(solution_3031(word="abacaba", k=3))
    print(solution_3031(word="abacaba", k=4))
    print(solution_3031(word="abcbabcd", k=2))


def testcase_3031_2():
    print(solution_3031_2(word="abacaba", k=3))
    print(solution_3031_2(word="abacaba", k=4))
    print(solution_3031_2(word="abcbabcd", k=2))


def testcase_3030():
    print(solution_3030(image=[[5, 6, 7, 10], [8, 9, 10, 10], [11, 12, 13, 10]], threshold=3))
    print(solution_3030(image=[[10, 20, 30], [15, 25, 35], [20, 30, 40], [25, 35, 45]], threshold=12))
    print(solution_3030(image=[[5, 6, 7], [8, 9, 10], [11, 12, 13]], threshold=1))


def testcase_1696():
    print(solution_1696(nums=[1, -1, -2, 4, -7, 3], k=2))
    print(solution_1696(nums=[10, -5, -2, 4, 0, 3], k=3))
    print(solution_1696(nums=[1, -5, -20, 4, -1, 3, -6, -3], k=2))


def testcase_1696_2():
    print(solution_1696_2(nums=[1, -1, -2, 4, -7, 3], k=2))
    print(solution_1696_2(nums=[10, -5, -2, 4, 0, 3], k=3))
    print(solution_1696_2(nums=[1, -5, -20, 4, -1, 3, -6, -3], k=2))


def testcase_lcp_30():
    print(solution_lcp_30(nums=[100, 100, 100, -250, -60, -140, -50, -50, 100, 150]))
    print(solution_lcp_30(nums=[-200, -300, 400, 0]))
    print(solution_lcp_30(nums=[-1, -1, 10]))


def testcase_3027():
    print(solution_3027([[1, 1], [2, 2], [3, 3]]))
    print(solution_3027([[6, 2], [4, 4], [2, 6]]))
    print(solution_3027([[3, 1], [1, 3], [1, 1]]))


def testcase_2641():
    # root=[5, 4, 9, 1, 10, null, 7]
    print(solution_2641(TreeNode(5, TreeNode(4), TreeNode(9))))
    print(solution_2641(TreeNode(5, TreeNode(4, TreeNode(1), TreeNode(10)), TreeNode(9, None, TreeNode(7)))))


def testcase_2641_2():
    # root=[5, 4, 9, 1, 10, null, 7]
    print(solution_2641_2(TreeNode(5, TreeNode(4), TreeNode(9))))
    print(solution_2641_2(TreeNode(5, TreeNode(4, TreeNode(1), TreeNode(10)), TreeNode(9, None, TreeNode(7)))))


def testcase_993():
    print(solution_993(root=TreeNode([1, 2, 3, 4, None, None, None]), x=4, y=3))
    print(solution_993(root=TreeNode([1, 2, 3, None, 4, None, 5]), x=5, y=4))
    print(solution_993(root=TreeNode([1, 2, 3, None, 4, None, None]), x=2, y=3))
    print(solution_993(root=TreeNode([1, 2, None, 3, 4, None, None, None, None, 5, None, None, None, None, None]), x=2,
                       y=4))


def testcase_236():
    q = TreeNode(4)
    p = TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), q))
    print(solution_236(TreeNode(3, p, TreeNode(1, TreeNode(0), TreeNode(8))), p, q))


def testcase_236_2():
    q = TreeNode(4)
    p = TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), q))
    print(solution_236_2(TreeNode(3, p, TreeNode(1, TreeNode(0), TreeNode(8))), p, q))


def testcase_236_3():
    q = TreeNode(4)
    p = TreeNode(5, TreeNode(6), TreeNode(2, TreeNode(7), q))
    print(solution_236_3(TreeNode(3, p, TreeNode(1, TreeNode(0), TreeNode(8))), p, q))


def testcase_3033():
    print(solution_3033([[1, 2, -1], [4, -1, 6], [7, 8, 9]]))
    print(solution_3033([[3, -1], [5, 2]]))


def testcase_3034():
    print(solution_3034(nums=[1, 2, 3, 4, 5, 6], pattern=[1, 1]))
    print(solution_3034(nums=[1, 4, 4, 1, 3, 5, 5, 3], pattern=[1, 0, -1]))


def testcase_3036():
    print(solution_3036(nums=[1, 2, 3, 4, 5, 6], pattern=[1, 1]))
    print(solution_3036(nums=[1, 4, 4, 1, 3, 5, 5, 3], pattern=[1, 0, -1]))


def testcase_100198_2():
    print(solution_100198_2(nums=[1, 2, 3, 4, 5, 6], pattern=[1, 1]))
    print(solution_100198_2(nums=[1, 4, 4, 1, 3, 5, 5, 3], pattern=[1, 0, -1]))


def testcase_987():
    print(solution_987(root=TreeNode([3, 9, 20, None, None, 15, 7])))
    print(solution_987(root=TreeNode([1, 2, 3, 4, 5, 6, 7])))
    print(solution_987(root=TreeNode([1, 2, 3, 4, 6, 5, 7])))


def testcase_102():
    print(solution_102(root=TreeNode([3, 9, 20, None, None, 15, 7])))
    print(solution_102(root=TreeNode([1, None, None])))
    print(solution_102(root=None))


def testcase_103():
    print(solution_103(root=TreeNode([3, 9, 20, None, None, 15, 7])))
    print(solution_103(root=TreeNode([1, None, None])))
    print(solution_103(root=None))


def testcase_107():
    print(solution_107(root=TreeNode([3, 9, 20, None, None, 15, 7])))
    print(solution_107(root=TreeNode([1, None, None])))
    print(solution_107(root=None))


def testcase_429():
    print(solution_429(root=Node([1, None, 3, 2, 4, None, 5, 6])))


def testcase_3038():
    print(solution_3038([3, 2, 1, 4, 5]))
    print(solution_3038([3, 2, 6, 1, 4]))


def testcase_3039():
    print(solution_3039("aabcbbca"))
    print(solution_3039("abcd"))


def testcase_3039_2():
    print(solution_3039_2("aabcbbca"))
    print(solution_3039_2("abcd"))


def testcase_3040():
    print(solution_3040([3, 2, 1, 2, 3, 4]))
    print(solution_3040([3, 2, 6, 1, 4]))


def testcase_3040_2():
    print(solution_3040_2([3, 2, 1, 2, 3, 4]))
    print(solution_3040_2([3, 2, 6, 1, 4]))


def testcase_3041():
    print(solution_3041([2, 1, 5, 1, 1]))
    print(solution_3041([1, 4, 7, 10]))
    print(solution_3041([8, 10, 6, 12, 9, 12, 2, 3, 13, 19, 11, 18, 10, 16]))


def testcase_3041_2():
    print(solution_3041_2([2, 1, 5, 1, 1]))
    print(solution_3041_2([1, 4, 7, 10]))
    print(solution_3041_2([8, 10, 6, 12, 9, 12, 2, 3, 13, 19, 11, 18, 10, 16]))
    print(solution_3041_2([12, 11, 8, 7, 2, 10, 18, 12]))


def testcase_589():
    print(solution_589(root=Node([1, None, 3, 2, 4, None, 5, 6])))


def testcase_589_2():
    print(solution_589_2(root=Node([1, None, 3, 2, 4, None, 5, 6])))


def testcase_3042():
    print(solution_3042(["a", "aba", "ababa", "aa"]))
    print(solution_3042(["pa", "papa", "ma", "mama"]))
    print(solution_3042(["abab", "ab"]))


def testcase_3043():
    print(solution_3043(arr1=[1, 10, 100], arr2=[1000]))
    print(solution_3043(arr1=[1, 2, 3], arr2=[4, 4, 4]))


def testcase_3044():
    print(solution_3044([[1, 1], [9, 9], [1, 1]]))
    print(solution_3044([[7]]))
    print(solution_3044([[9, 7, 8], [4, 6, 5], [2, 8, 6]]))


def testcase_590():
    print(solution_590(Node([1, None, 3, 2, 4, None, 5, 6])))
    print(solution_590(Node(
        [1, None, 2, 3, 4, 5, None, None, 6, 7, None, 8, None, 9, 10, None, None, 11, None, 12, None, 13, None, None,
         14])))


def testcase_590_2():
    print(solution_590_2(Node([1, None, 3, 2, 4, None, 5, 6])))
    print(solution_590_2(Node(
        [1, None, 2, 3, 4, 5, None, None, 6, 7, None, 8, None, 9, 10, None, None, 11, None, 12, None, 13, None, None,
         14])))


def testcase_105():
    print(solution_105([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]))
    print(solution_105([-1], [-1]))
    print(solution_105([1, 2], [2, 1]))


def testcase_105_2():
    print(solution_105_2([3, 9, 20, 15, 7], [9, 3, 15, 20, 7]))
    print(solution_105_2([-1], [-1]))
    print(solution_105_2([1, 2], [2, 1]))


def testcase_106():
    print(solution_106([9, 3, 15, 20, 7], [9, 15, 7, 20, 3]))
    print(solution_106([-1], [-1]))
    print(solution_106([1, 2], [2, 1]))


def testcase_106_2():
    print(solution_106_2([9, 3, 15, 20, 7], [9, 15, 7, 20, 3]))
    print(solution_106_2([-1], [-1]))
    print(solution_106_2([1, 2], [2, 1]))


def testcase_3045():
    print(solution_3045(["a", "a"]))
    print(solution_3045(["a", "aba", "ababa", "aa"]))
    print(solution_3045(["pa", "papa", "ma", "mama"]))
    print(solution_3045(["abab", "ab"]))


def testcase_889():
    print(solution_889([1, 2, 4, 5, 3, 6, 7], [4, 5, 2, 6, 7, 3, 1]))


def testcase_2583():
    print(solution_2583(TreeNode([5, 8, 9, 2, 1, 3, 7, 4, 6, None, None, None, None, None, None]), 2))
    print(solution_2583(TreeNode([1, 2, None, 3, None, None, None]), 1))


def testcase_2476():
    print(solution_2476(TreeNode([6, 2, 13, 1, 4, 9, 15, None, None, None, None, None, None, 14, None]), [2, 5, 16]))
    print(solution_2476(TreeNode([4, None, 9]), [3]))


def testcase_2476_2():
    print(solution_2476_2(TreeNode([6, 2, 13, 1, 4, 9, 15, None, None, None, None, None, None, 14, None]), [2, 5, 16]))
    print(solution_2476_2(TreeNode([4, None, 9]), [3]))


def testcase_235():
    p = TreeNode(2, TreeNode(0), TreeNode(4, TreeNode(3), TreeNode(5)))
    q = TreeNode(8, TreeNode(7), TreeNode(9))
    print(solution_235(TreeNode(6, p, q), p, q))


def testcase_235_2():
    p = TreeNode(2, TreeNode(0), TreeNode(4, TreeNode(3), TreeNode(5)))
    q = TreeNode(8, TreeNode(7), TreeNode(9))
    print(solution_235_2(TreeNode(6, p, q), p, q))


def testcase_3046():
    print(solution_3046([1, 1, 2, 2, 3, 4]))
    print(solution_3046([1, 1, 1, 1]))


def testcase_3047():
    print(solution_3047(bottomLeft=[[1, 1], [2, 2], [3, 1]], topRight=[[3, 3], [4, 4], [6, 6]]))
    print(solution_3047(bottomLeft=[[1, 1], [2, 2], [1, 2]], topRight=[[3, 3], [4, 4], [3, 4]]))
    print(solution_3047(bottomLeft=[[1, 1], [3, 3], [3, 1]], topRight=[[2, 2], [4, 4], [4, 2]]))
    print(solution_3047(bottomLeft=[[1, 2], [1, 2]], topRight=[[4, 5], [2, 3]]))
    print(solution_3047([[2, 2], [1, 3]], [[3, 4], [5, 5]]))
    print(solution_3047([[2, 2], [3, 1]], [[5, 5], [5, 5]]))


def testcase_938():
    print(solution_938(TreeNode([10, 5, 15, 3, 7, None, 18]), 7, 15))
    print(solution_938(TreeNode([10, 5, 15, 3, 7, 13, 18, 1, None, 6]), 6, 10))


def testcase_2867():
    print(solution_2867(n=5, edges=[[1, 2], [1, 3], [2, 4], [2, 5]]))
    print(solution_2867(n=6, edges=[[1, 2], [1, 3], [2, 4], [3, 5], [3, 6]]))


def testcase_2673():
    print(solution_2673(7, [1, 5, 2, 2, 3, 3, 1]))
    print(solution_2673(3, [5, 3, 3]))


def testcase_2581():
    print(solution_2581(edges=[[0, 1], [1, 2], [1, 3], [4, 2]], guesses=[[1, 3], [0, 1], [1, 0], [2, 4]], k=3))
    print(solution_2581(edges=[[0, 1], [1, 2], [2, 3], [3, 4]], guesses=[[1, 0], [3, 4], [2, 1], [3, 2]], k=1))


def testcase_2369():
    print(solution_2369([4, 4, 4, 5, 6]))
    print(solution_2369([1, 1, 1, 2]))
    print(solution_2369([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))


def testcase_2369_2():
    print(solution_2369_2([4, 4, 4, 5, 6]))
    print(solution_2369_2([1, 1, 1, 2]))
    print(solution_2369_2([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))


def testcase_3048():
    print(solution_3048(nums=[2, 2, 0], changeIndices=[2, 2, 2, 2, 3, 2, 2, 1]))
    print(solution_3048(nums=[1, 3], changeIndices=[1, 1, 1, 2, 1, 1, 1]))
    print(solution_3048(nums=[0, 1], changeIndices=[2, 2, 2]))


def testcase_2368():
    print(solution_2368(n=7, edges=[[0, 1], [1, 2], [3, 1], [4, 0], [0, 5], [5, 6]], restricted=[4, 5]))
    print(solution_2368(n=7, edges=[[0, 1], [0, 2], [0, 5], [0, 4], [3, 2], [6, 5]], restricted=[4, 2, 1]))


def testcase_3049():
    print(solution_3049(nums=[3, 2, 3], changeIndices=[1, 3, 2, 2, 2, 2, 3]))
    print(solution_3049(nums=[0, 0, 1, 2], changeIndices=[1, 2, 1, 2, 1, 2, 1, 2]))
    print(solution_3049(nums=[1, 2, 3], changeIndices=[1, 2, 3]))


def testcase_3065():
    print(solution_3065([2, 11, 10, 1, 3], 10))
    print(solution_3065(nums=[1, 1, 2, 4, 9], k=1))
    print(solution_3065(nums=[1, 1, 2, 4, 9], k=9))


def testcase_3066():
    print(solution_3066(nums=[2, 11, 10, 1, 3], k=10))
    print(solution_3066(nums=[1, 1, 2, 4, 9], k=20))


def testcase_3067():
    print(
        solution_3067(edges=[[0, 1, 1], [1, 2, 5], [2, 3, 13], [3, 4, 9], [4, 5, 2]], signalSpeed=1))


def testcase_3067_2():
    print(
        solution_3067_2(edges=[[0, 1, 1], [1, 2, 5], [2, 3, 13], [3, 4, 9], [4, 5, 2]], signalSpeed=1))


def testcase_3069():
    print(solution_3069(nums=[2, 1, 3]))
    print(solution_3069(nums=[5, 4, 3, 8]))


def testcase_3070():
    print(solution_3070(grid=[[7, 6, 3], [6, 6, 1]], k=18))
    print(solution_3070(grid=[[7, 2, 9], [1, 5, 0], [2, 6, 6]], k=20))


def testcase_3071():
    print(solution_3071(grid=[[1, 2, 2], [1, 1, 0], [0, 1, 0]]))
    print(solution_3071(
        grid=[[0, 1, 0, 1, 0], [2, 1, 0, 1, 2], [2, 2, 2, 0, 1], [2, 2, 2, 2, 2], [2, 1, 2, 2, 2]]))


def testcase_3072():
    # print(solution_3072())
    ...


def testcase_1976():
    print(solution_1976(n=7,
                        roads=[[0, 6, 7], [0, 1, 2], [1, 2, 3], [1, 3, 3], [6, 3, 3], [3, 5, 1], [6, 5, 1], [2, 5, 1],
                               [0, 4, 5], [4, 6, 2]]))
    print(solution_1976(n=2, roads=[[1, 0, 10]]))
    print(solution_1976(n=6,
                        roads=[[3, 0, 4], [0, 2, 3], [1, 2, 2], [4, 1, 3], [2, 5, 5], [2, 3, 1], [0, 4, 1], [2, 4, 6],
                               [4, 3, 1]]))


def testcase_1976_2():
    print(solution_1976_2(n=7,
                          roads=[[0, 6, 7], [0, 1, 2], [1, 2, 3], [1, 3, 3], [6, 3, 3], [3, 5, 1], [6, 5, 1], [2, 5, 1],
                                 [0, 4, 5], [4, 6, 2]]))
    print(solution_1976_2(n=2, roads=[[1, 0, 10]]))
    print(solution_1976_2(n=6,
                          roads=[[3, 0, 4], [0, 2, 3], [1, 2, 2], [4, 1, 3], [2, 5, 5], [2, 3, 1], [0, 4, 1], [2, 4, 6],
                                 [4, 3, 1]]))


def testcase_1976_3():
    print(solution_1976_3(n=7,
                          roads=[[0, 6, 7], [0, 1, 2], [1, 2, 3], [1, 3, 3], [6, 3, 3], [3, 5, 1], [6, 5, 1], [2, 5, 1],
                                 [0, 4, 5], [4, 6, 2]]))
    print(solution_1976_3(n=2, roads=[[1, 0, 10]]))
    print(solution_1976_3(n=6,
                          roads=[[3, 0, 4], [0, 2, 3], [1, 2, 2], [4, 1, 3], [2, 5, 5], [2, 3, 1], [0, 4, 1], [2, 4, 6],
                                 [4, 3, 1]]))


def testcase_2917():
    print(solution_2917(nums=[7, 12, 9, 8, 9, 15], k=4))
    print(solution_2917(nums=[2, 12, 1, 11, 4, 5], k=6))
    print(solution_2917(nums=[10, 8, 5, 9, 11, 6, 8], k=1))


def testcase_257():
    print(solution_257(root=TreeNode([1, 2, 3, None, 5])))
    print(solution_257(root=TreeNode(1)))


def testcase_258():
    print(solution_258(38))
    print(solution_258(0))
    print(solution_258(10))


def testcase_258_2():
    print(solution_258_2(38))
    print(solution_258_2(0))
    print(solution_258_2(10))


def testcase_3068():
    print(solution_3068(nums=[1, 2, 1], k=3, edges=[[0, 1], [0, 2]]))
    print(solution_3068(nums=[2, 3], k=7, edges=[[0, 1]]))
    print(solution_3068(nums=[7, 7, 7, 7, 7, 7], k=3, edges=[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]))


def testcase_3068_2():
    print(solution_3068_2(nums=[1, 2, 1], k=3, edges=[[0, 1], [0, 2]]))
    print(solution_3068_2(nums=[2, 3], k=7, edges=[[0, 1]]))
    print(solution_3068_2(nums=[7, 7, 7, 7, 7, 7], k=3, edges=[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]))


def testcase_3068_3():
    print(solution_3068_3(nums=[1, 2, 1], k=3, edges=[[0, 1], [0, 2]]))
    print(solution_3068_3(nums=[2, 3], k=7, edges=[[0, 1]]))
    print(solution_3068_3(nums=[7, 7, 7, 7, 7, 7], k=3, edges=[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]))


def testcase_3068_4():
    print(solution_3068_4(nums=[1, 2, 1], k=3, edges=[[0, 1], [0, 2]]))
    print(solution_3068_4(nums=[2, 3], k=7, edges=[[0, 1]]))
    print(solution_3068_4(nums=[7, 7, 7, 7, 7, 7], k=3, edges=[[0, 1], [0, 2], [0, 3], [0, 4], [0, 5]]))


def testcase_2575():
    print(solution_2575(word="998244353", m=3))
    print(solution_2575(word="1010", m=10))


def testcase_263():
    print(solution_263(6))
    print(solution_263(1))
    print(solution_263(14))


def testcase_268():
    print(solution_268([3, 0, 1]))
    print(solution_268([0, 1]))
    print(solution_268([9, 6, 4, 2, 3, 5, 7, 0, 1]))
    print(solution_268([0]))


def testcase_15():
    print(solution_15([-1, 0, 1, 2, -1, -4]))
    print(solution_15([0, 1, 1]))
    print(solution_15([0, 0, 0]))


def testcase_15_2():
    print(solution_15_2([-1, 0, 1, 2, -1, -4]))
    print(solution_15_2([0, 1, 1]))
    print(solution_15_2([0, 0, 0]))


def testcase_167():
    print(solution_167(numbers=[2, 7, 11, 15], target=9))
    print(solution_167(numbers=[2, 3, 4], target=6))
    print(solution_167(numbers=[-1, 0], target=-1))


def testcase_18():
    print(solution_18(nums=[1, 0, -1, 0, -2, 2], target=0))
    print(solution_18(nums=[2, 2, 2, 2, 2], target=8))


def testcase_18_2():
    print(solution_18_2(nums=[1, 0, -1, 0, -2, 2], target=0))
    print(solution_18_2(nums=[2, 2, 2, 2, 2], target=8))


def testcase_2834():
    print(solution_2834(n=2, target=3))
    print(solution_2834(n=3, target=3))
    print(solution_2834(n=1, target=1))
    print(solution_2834(n=5, target=4))
    print(solution_2834(n=13, target=50))
    print(solution_2834(n=39636, target=49035))


def testcase_278():
    def getisBadVersion(fail_version: int) -> Callable:
        def isBadVersion(version: int) -> bool:
            return version >= fail_version

        return isBadVersion

    print(solution_278(5, getisBadVersion(4)))
    print(solution_278(1, getisBadVersion(1)))
    print(solution_278(30, getisBadVersion(5)))


def testcase_283():
    a = [0, 1, 0, 3, 12]
    solution_283(a)
    print(a)
    a = [0]
    solution_283(a)
    print(a)
    a = [1, 0]
    solution_283(a)
    print(a)
    a = [1, 2, 3, 0, 4, 5, 0, 0, 0]
    solution_283(a)
    print(a)


def testcase_303():
    obj = NumArray(nums=[-2, 0, 3, -5, 2, -1])
    print(obj.sumRange(0, 2))
    print(obj.sumRange(2, 5))
    print(obj.sumRange(0, 5))


def testcase_326():
    print(solution_326(27))
    print(solution_326(0))
    print(solution_326(9))
    print(solution_326(45))


def testcase_11():
    print(solution_11([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print(solution_11([1, 1]))


def testcase_42():
    print(solution_42([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(solution_42([4, 2, 0, 3, 2, 5]))


def testcase_42_2():
    print(solution_42_2([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(solution_42_2([4, 2, 0, 100, 2, 5]))


def testcase_42_3():
    print(solution_42_3([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))
    print(solution_42_3([4, 2, 0, 100, 2, 5]))


def testcase_2386():
    print(solution_2386(nums=[2, 4, -2], k=5))
    print(solution_2386(nums=[1, -2, 3, 4, -10, 12], k=16))


def testcase_2386_2():
    print(solution_2386_2(nums=[2, 4, -2], k=5))
    print(solution_2386_2(nums=[1, -2, 3, 4, -10, 12], k=16))


def testcase_299():
    print(solution_299("1807", "7810"))
    print(solution_299("1123", "0111"))


def testcase_3074():
    print(solution_3074(apple=[1, 3, 2], capacity=[4, 3, 1, 5, 2]))
    print(solution_3074(apple=[5, 5, 5], capacity=[2, 4, 2, 7]))


def testcase_3075():
    print(solution_3075(happiness=[1, 2, 3], k=2))
    print(solution_3075(happiness=[1, 1, 1, 1], k=2))
    print(solution_3075(happiness=[2, 3, 4, 5], k=1))


def testcase_3076():
    print(solution_3076(arr=["cab", "ad", "bad", "c"]))
    print(solution_3076(arr=["abc", "bcd", "abcd"]))
    print(solution_3076(arr=["vbb", "grg", "lexn", "oklqe", "yxav"]))


def testcase_2129_0():
    print("Capitalize The Title")
    print("First Letter of Each Word")
    print("i Love Leetcode")
    print("l hv")


def testcase_2129():
    print(solution_2129("capiTalIze tHe titLe"))
    print(solution_2129("First leTTeR of EACH Word"))
    print(solution_2129("i lOve leetcode"))
    print(solution_2129_2("L hv"))


def testcase_2129_2():
    print(solution_2129_2("capiTalIze tHe titLe"))
    print(solution_2129_2("First leTTeR of EACH Word"))
    print(solution_2129_2("i lOve leetcode"))
    print(solution_2129_2("L hv"))


def testcase_3077_0():
    print(22)
    print(64)
    print(-1)


def testcase_3077():
    print(solution_3077(nums=[1, 2, 3, -1, 2], k=3))
    print(solution_3077(nums=[12, -2, -2, -2, -2], k=5))
    print(solution_3077(nums=[-1, -2, -3], k=1))


def testcase_338():
    print(solution_338(2))
    print(solution_338(5))


def testcase_338_2():
    print(solution_338_2(2))
    print(solution_338_2(5))


def testcase_1261():
    root = TreeNode([-1, None, -1])
    obj = FindElements(root)
    print(obj.find(1))
    print(obj.find(2))
    root = TreeNode([-1, -1, -1, -1, -1])
    obj = FindElements(root)
    print(obj.find(1))
    print(obj.find(3))
    print(obj.find(5))
    root = TreeNode([-1, None, -1, None, None, -1, None, None, None, None, None, -1])
    obj = FindElements(root)
    print(obj.find(2))
    print(obj.find(3))
    print(obj.find(4))
    print(obj.find(5))


def testcase_342():
    print(solution_342(16))
    print(solution_342(5))
    print(solution_342(1))


def testcase_344():
    s = ["h", "e", "l", "l", "o"]
    solution_344(s)
    print(s)
    s = ["H", "a", "n", "n", "a", "h"]
    solution_344(s)
    print(s)


def testcase_345():
    print(solution_345('hello'))
    print(solution_345('leetcode'))


def testcase_2864():
    print(solution_2864('010'))
    print(solution_2864('0101'))


def testcase_349():
    print(solution_349([1, 2, 2, 1], [2, 2]))
    print(solution_349([4, 9, 5], [9, 4, 9, 8, 4]))


def testcase_350():
    print(solution_350([1, 2, 2, 1], [2, 2]))
    print(solution_350([4, 4, 9, 5], [9, 4, 9, 8, 4]))


def testcase_350_1():
    print(solution_350_1([1, 2, 2, 1], [2, 2]))
    print(solution_350_1([4, 4, 9, 5], [9, 4, 9, 8, 4]))


def testcase_367():
    print(solution_367(16))
    print(solution_367(14))


def testcase_367_1():
    print(solution_367_1(16))
    print(solution_367_1(14))


def testcase_367_2():
    print(solution_367_2(16))
    print(solution_367_2(14))


def testcase_2789():
    print(solution_2789([2, 3, 7, 9, 3]))
    print(solution_2789([5, 3, 3]))


def testcase_374():
    def get_guess(pick: int) -> Callable:
        def guess(n: int) -> int:
            if n == pick:
                return 0
            elif n > pick:
                return -1
            else:
                return 1

        return guess

    print(solution_374(10, get_guess(6)))
    print(solution_374(1, get_guess(1)))
    print(solution_374(2, get_guess(1)))
    print(solution_374(2, get_guess(2)))


def testcase_374_2():
    def get_guess(pick: int) -> Callable:
        def guess(n: int) -> int:
            if n == pick:
                return 0
            elif n > pick:
                return -1
            else:
                return 1

        return guess

    print(solution_374_2(10, get_guess(6)))
    print(solution_374_2(1, get_guess(1)))
    print(solution_374_2(2, get_guess(1)))
    print(solution_374_2(2, get_guess(2)))


def testcase_387():
    print(solution_387("leetcode"))
    print(solution_387("loveleetcode"))
    print(solution_387("aabb"))


def testcase_2312():
    print(solution_2312(m=3, n=5, prices=[[1, 4, 2], [2, 2, 7], [2, 1, 3]]))
    print(solution_2312(m=4, n=6, prices=[[3, 2, 10], [1, 4, 2], [4, 1, 3]]))


def testcase_2312_2():
    print(solution_2312_2(m=3, n=5, prices=[[1, 4, 2], [2, 2, 7], [2, 1, 3]]))
    print(solution_2312_2(m=4, n=6, prices=[[3, 2, 10], [1, 4, 2], [4, 1, 3]]))


def testcase_389():
    print(solution_389("abcd", "abcde"))
    print(solution_389("", "y"))
    print(solution_389("a", "aa"))


def testcase_392():
    print(solution_392(s="abc", t="ahbgdc"))
    print(solution_392(s="axc", t="ahbgdc"))


def testcase_2684():
    print(solution_2684(grid=[[2, 4, 3, 5], [5, 4, 9, 3], [3, 4, 2, 11], [10, 9, 13, 15]]))
    print(solution_2684(grid=[[3, 2, 4], [2, 1, 9], [1, 1, 7]]))


def testcase_3079():
    print(solution_3079([1, 2, 3]))
    print(solution_3079([10, 21, 31]))


def testcase_3079_2():
    print(solution_3079_2([1, 2, 3]))
    print(solution_3079_2([10, 21, 31]))


def testcase_3080():
    print(solution_3080([1, 2, 2, 1, 2, 3, 1], queries=[[1, 2], [3, 3], [4, 2]]))
    print(solution_3080([1, 4, 2, 3], queries=[[1, 1], [1, 1]]))


def testcase_3081():
    print(solution_3081("???"))
    print(solution_3081("a?a?"))


def testcase_3082():
    print(solution_3082(nums=[1, 2, 3], k=3))
    print(solution_3082(nums=[2, 3, 3], k=5))
    print(solution_3082(nums=[1, 2, 3], k=7))


def testcase_3082_2():
    print(solution_3082_2(nums=[1, 2, 3], k=3))
    print(solution_3082_2(nums=[2, 3, 3], k=5))
    print(solution_3082_2(nums=[1, 2, 3], k=7))


def testcase_310():
    print(solution_310(n=4, edges=[[1, 0], [1, 2], [1, 3]]))
    print(solution_310(n=6, edges=[[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]]))
    print(solution_310(n=6, edges=[[0, 1], [0, 2], [0, 3], [3, 4], [4, 5]]))


def testcase_310_2():
    print(solution_310_2(n=4, edges=[[1, 0], [1, 2], [1, 3]]))
    print(solution_310_2(n=6, edges=[[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]]))
    print(solution_310_2(n=6, edges=[[0, 1], [0, 2], [0, 3], [3, 4], [4, 5]]))


def testcase_3083():
    print(solution_3083(s="leetcode"))
    print(solution_3083(s="abcba"))
    print(solution_3083(s="abcd"))


def testcase_3083_2():
    print(solution_3083_2(s="leetcode"))
    print(solution_3083_2(s="abcba"))
    print(solution_3083_2(s="abcd"))


def testcase_3084():
    print(solution_3084(s="abada", c="a"))
    print(solution_3084(s="zzz", c="z"))
    print(solution_3084(s="aaaa", c="a"))


def testcase_3085():
    print(solution_3085(word="aabcaba", k=0))
    print(solution_3085(word="dabdcbdcdcd", k=2))
    print(solution_3085(word="aaabaaa", k=2))


def testcase_3085_2():
    print(solution_3085_2(word="aabcaba", k=0))
    print(solution_3085_2(word="dabdcbdcdcd", k=2))
    print(solution_3085_2(word="aaabaaa", k=2))


def testcase_3086():
    print(solution_3086(nums=[1, 1, 0, 0, 0, 1, 1, 0, 0, 1], k=3, maxChanges=1))
    print(solution_3086(nums=[0, 0, 0, 0], k=2, maxChanges=3))


def testcase_1793():
    print(solution_1793([1, 4, 3, 7, 4, 5], 3))
    print(solution_1793(nums=[5, 5, 4, 5, 4, 1, 1, 1], k=0))


def testcase_1969():
    print(solution_1969(1))
    print(solution_1969(2))
    print(solution_1969(3))
    print(solution_1969(4))


def testcase_401():
    print(solution_401(1))
    print(solution_401(9))


def testcase_404():
    print(solution_404(TreeNode([3, 9, 20, None, None, 15, 7])))
    print(solution_404(TreeNode(1)))


def testcase_665():
    print(solution_665([4, 2, 3]))
    print(solution_665([4, 2, 1]))


def testcase_2671():
    ft = FrequencyTracker()
    ft.add(3)
    ft.add(3)
    print(ft.hasFrequency(2))
    ft = FrequencyTracker()
    ft.add(1)
    ft.deleteOne(1)
    print(ft.hasFrequency(1))
    ft = FrequencyTracker()
    print(ft.hasFrequency(2))
    ft.add(3)
    print(ft.hasFrequency(1))


def testcase_2617():
    print(solution_2617([[2, 1, 0], [1, 0, 0]]))
    print(solution_2617([[3, 4, 2, 1], [4, 2, 1, 1], [2, 1, 1, 0], [3, 4, 1, 0]]))
    print(solution_2617([[3, 4, 2, 1], [4, 2, 3, 1], [2, 1, 0, 0], [2, 4, 0, 0]]))


def testcase_2617_2():
    print(solution_2617_2([[2, 1, 0], [1, 0, 0]]))
    print(solution_2617_2([[3, 4, 2, 1], [4, 2, 1, 1], [2, 1, 1, 0], [3, 4, 1, 0]]))
    print(solution_2617_2([[3, 4, 2, 1], [4, 2, 3, 1], [2, 1, 0, 0], [2, 4, 0, 0]]))


def testcase_2617_3():
    print(solution_2617_3([[2, 1, 0], [1, 0, 0]]))
    print(solution_2617_3([[3, 4, 2, 1], [4, 2, 1, 1], [2, 1, 1, 0], [3, 4, 1, 0]]))
    print(solution_2617_3([[3, 4, 2, 1], [4, 2, 3, 1], [2, 1, 0, 0], [2, 4, 0, 0]]))


def testcase_2549():
    print(solution_2549(5))
    print(solution_2549(1))


def testcase_322():
    print(solution_322(coins=[186, 419, 83, 408], amount=6249))
    print(solution_322(coins=[1, 2, 5], amount=11))
    print(solution_322(coins=[2], amount=3))
    print(solution_322(coins=[1], amount=0))


def testcase_322_2():
    print(solution_322_2(coins=[186, 419, 83, 408], amount=6249))
    print(solution_322_2(coins=[1, 2, 5], amount=11))
    print(solution_322_2(coins=[2], amount=3))
    print(solution_322_2(coins=[1], amount=0))


def testcase_3090():
    print(solution_3090("bcbbbcba"))
    print(solution_3090("aaaa"))


def testcase_3090_2():
    print(solution_3090_2("bcbbbcba"))
    print(solution_3090_2("aaaa"))


def testcase_3091():
    print(solution_3091(11))
    print(solution_3091(1))


def testcase_3092():
    print(solution_3092(nums=[2, 3, 2, 1], freq=[3, 2, -3, 1]))
    print(solution_3092(nums=[5, 5, 3], freq=[2, -2, 1]))


def testcase_3092_2():
    print(solution_3092_2(nums=[2, 3, 2, 1], freq=[3, 2, -3, 1]))
    print(solution_3092_2(nums=[5, 5, 3], freq=[2, -2, 1]))


def testcase_3093():
    print(solution_3093(wordsContainer=["abcd", "bcd", "xbcd"], wordsQuery=["cd", "bcd", "xyz"]))
    print(solution_3093(wordsContainer=["abcdefgh", "poiuygh", "ghghgh"], wordsQuery=["gh", "acbfgh", "acbfegh"]))


def testcase_518():
    print(solution_518(5, [1, 2, 5]))
    print(solution_518(3, [2]))
    print(solution_518(10, [10]))


def testcase_518_2():
    print(solution_518_2(5, [1, 2, 5]))
    print(solution_518_2(3, [2]))
    print(solution_518_2(10, [10]))


def testcase_2642():
    g = Graph(4, [[0, 2, 5], [0, 1, 2], [1, 2, 1], [3, 0, 3]])
    print(g.shortestPath(3, 2))
    print(g.shortestPath(0, 3))
    g.addEdge([1, 3, 4])
    print(g.shortestPath(0, 3))


def testcase_2580():
    print(solution_2580(ranges=[[6, 10], [5, 15]]))
    print(solution_2580(ranges=[[1, 3], [10, 20], [2, 5], [4, 8]]))


def testcase_704():
    print(solution_704(nums=[-1, 0, 3, 5, 9, 12], target=9))
    print(solution_704(nums=[-1, 0, 3, 5, 9, 12], target=2))


def testcase_1997():
    print(solution_1997([0, 0, 2]))
    print(solution_1997([0, 0]))
    print(solution_1997([0, 1, 2, 0]))
    print(solution_1997([0, 0, 0, 1, 4]))
    print(solution_1997([0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1, 8]))
    print(solution_1997(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0]))


def testcase_2908():
    print(solution_2908([8, 6, 1, 5, 3]))
    print(solution_2908([5, 4, 8, 7, 10, 2]))
    print(solution_2908([6, 5, 4, 3, 4, 5]))


def testcase_1004():
    print(solution_1004(nums=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], k=2))
    print(solution_1004(nums=[0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], k=3))


def testcase_2952():
    print(solution_2952(coins=[1, 4, 10], target=19))
    print(solution_2952(coins=[1, 4, 10, 5, 7, 19], target=19))
    print(solution_2952(coins=[1, 1, 1], target=20))


def testcase_3095():
    print(solution_3095(nums=[1, 2, 3], k=2))
    print(solution_3095(nums=[2, 1, 8], k=10))
    print(solution_3095(nums=[1, 2], k=0))
    print(solution_3095(nums=[32, 1, 25, 11, 2], k=59))


def testcase_3096():
    print(solution_3096(possible=[1, 1]))
    print(solution_3096(possible=[1, 0, 1, 0]))
    print(solution_3096(possible=[1, 1, 1, 1, 1]))
    print(solution_3096(possible=[0, 0]))


def testcase_3096_2():
    print(solution_3096_2(possible=[1, 1]))
    print(solution_3096_2(possible=[1, 0, 1, 0]))
    print(solution_3096_2(possible=[1, 1, 1, 1, 1]))
    print(solution_3096_2(possible=[0, 0]))


def testcase_3097():
    print(solution_3097(nums=[1, 2, 3], k=2))
    print(solution_3097(nums=[2, 1, 8], k=10))
    print(solution_3097(nums=[1, 2], k=0))
    print(solution_3097(nums=[32, 1, 25, 11, 2], k=59))
    print(solution_3097(nums=[1, 2, 12, 1, 16, 10], k=20))


def testcase_3097_2():
    print(solution_3097_2(nums=[1, 2, 3], k=2))
    print(solution_3097_2(nums=[2, 1, 8], k=10))
    print(solution_3097_2(nums=[1, 2], k=0))
    print(solution_3097_2(nums=[32, 1, 25, 11, 2], k=59))
    print(solution_3097_2(nums=[1, 2, 12, 1, 16, 10], k=20))


def testcase_3098():
    print(solution_3098(nums=[1, 2, 3, 4], k=3))
    print(solution_3098(nums=[2, 2], k=2))
    print(solution_3098(nums=[4, 3, -1], k=2))


def testcase_3099():
    print(solution_3099(18))
    print(solution_3099(23))


def testcase_3100():
    print(solution_3100(numBottles=13, numExchange=6))
    print(solution_3100(numBottles=10, numExchange=3))


def testcase_3101():
    print(solution_3101(nums=[0, 1, 1, 1]))
    print(solution_3101(nums=[1, 0, 1, 0]))


def testcase_3102():
    print(solution_3102(points=[[3, 10], [5, 15], [10, 2], [4, 4]]))
    print(solution_3102(points=[[1, 1], [1, 1], [1, 1]]))


def testcase_331():
    print(solution_331(preorder="9,3,4,#,#,1,#,#,2,#,6,#,#"))
    print(solution_331(preorder="1,#"))
    print(solution_331(preorder="9,#,#,1"))


def testcase_331_2():
    print(solution_331_2(preorder="9,3,4,#,#,1,#,#,2,#,6,#,#"))
    print(solution_331_2(preorder="1,#"))
    print(solution_331_2(preorder="9,#,#,1"))


def testcase_2810():
    print(solution_2810("string"))
    print(solution_2810("poiinter"))


def testcase_2810_2():
    print(solution_2810_2("string"))
    print(solution_2810_2("poiinter"))


def testcase_894():
    print(solution_894(3))
    ans = solution_894(7)
    print(ans)


def testcase_1379():
    target = TreeNode(3, TreeNode(6), TreeNode(19))
    target_c = TreeNode(3, TreeNode(6), TreeNode(19))
    print(solution_1379(TreeNode(7, TreeNode(4), target), TreeNode(7, TreeNode(4), target_c), target))


def testcase_1379_2():
    target = TreeNode(3, TreeNode(6), TreeNode(19))
    target_c = TreeNode(3, TreeNode(6), TreeNode(19))
    print(solution_1379_2(TreeNode(7, TreeNode(4), target), TreeNode(7, TreeNode(4), target_c), target))


def testcase_209():
    print(solution_209(7, [2, 3, 1, 2, 4, 3]))
    print(solution_209(4, [1, 4, 4]))
    print(solution_209(11, [1, 1, 1, 1, 1, 1, 1, 1]))


def testcase_2192_0():
    print("[[], [], [], [0, 1], [0, 2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3]]")
    print("[[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]")
    print("[[2, 4, 5], [0, 2, 4, 5], [4], [0, 1, 2, 4, 5], [], [2, 4]]")
    print("[[], [0, 2, 3, 6, 7], [], [0, 2, 6, 7], [0, 1, 2, 3, 5, 6, 7], [0, 1, 2, 3, 6, 7], [0, 2, 7], [0, 2]]")
    print("[[4], [0, 2, 3, 4, 5], [0, 4, 5], [0, 2, 4, 5], [], []]")


def testcase_2192():
    print(solution_2192(n=8, edges=[[0, 3], [0, 4], [1, 3], [2, 4], [2, 7], [3, 5], [3, 6], [3, 7], [4, 6]]))
    print(solution_2192(n=5, edges=[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]))
    print(solution_2192(n=6,
                        edges=[[0, 3], [5, 0], [2, 3], [4, 3], [5, 3], [1, 3], [2, 5], [0, 1], [4, 5], [4, 2], [4, 0],
                               [2, 1], [5, 1]]))
    print(solution_2192(n=8,
                        edges=[[0, 7], [7, 6], [0, 3], [6, 3], [5, 4], [1, 5], [2, 7], [3, 5], [3, 1], [0, 5], [7, 5],
                               [2, 1], [1, 4], [6, 1]]))
    print(solution_2192(n=6, edges=[[5, 1], [2, 3], [5, 3], [0, 2], [3, 1], [5, 2], [4, 0]]))


def testcase_2192_2():
    print(solution_2192_2(n=8, edges=[[0, 3], [0, 4], [1, 3], [2, 4], [2, 7], [3, 5], [3, 6], [3, 7], [4, 6]]))
    print(solution_2192_2(n=5, edges=[[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]))
    print(solution_2192_2(n=6,
                          edges=[[0, 3], [5, 0], [2, 3], [4, 3], [5, 3], [1, 3], [2, 5], [0, 1], [4, 5], [4, 2], [4, 0],
                                 [2, 1], [5, 1]]))
    print(solution_2192_2(n=8,
                          edges=[[0, 7], [7, 6], [0, 3], [6, 3], [5, 4], [1, 5], [2, 7], [3, 5], [3, 1], [0, 5], [7, 5],
                                 [2, 1], [1, 4], [6, 1]]))
    print(solution_2192_2(n=6, edges=[[5, 1], [2, 3], [5, 3], [0, 2], [3, 1], [5, 2], [4, 0]]))


def testcase_1026():
    print(solution_1026(TreeNode([8, 3, 10, 1, 6, None, 14, None, None, 4, 7, 13])))
    print(solution_1026(TreeNode([1, None, 2, None, None, None, 0, None, None, None, None, None, None, 3, None])))


def testcase_1600():
    t = ThroneInheritance("king")
    t.birth("king", "andy")
    t.birth("king", "bob")
    t.birth("king", "catherine")
    t.birth("andy", "matthew")
    t.birth("bob", "alex")
    t.birth("bob", "asha")
    print(t.getInheritanceOrder())
    t.death("bob")
    print(t.getInheritanceOrder())


def testcase_3105():
    print(solution_3105([1, 4, 3, 3, 2]))
    print(solution_3105([3, 3, 3, 3]))
    print(solution_3105([3, 2, 1]))
    print(solution_3105([1, 2]))


def testcase_3105_2():
    print(solution_3105_2([1, 4, 3, 3, 2]))
    print(solution_3105_2([3, 3, 3, 3]))
    print(solution_3105_2([3, 2, 1]))
    print(solution_3105_2([1, 2]))


def testcase_3106():
    print(solution_3106(s="zbbz", k=3))
    print(solution_3106(s="xaxcd", k=4))
    print(solution_3106(s="lol", k=0))


def testcase_3107():
    print(solution_3107(nums=[2, 5, 6, 8, 5], k=4))
    print(solution_3107(nums=[2, 5, 6, 8, 5], k=7))
    print(solution_3107(nums=[1, 2, 3, 4, 5, 6], k=4))


def testcase_3108():
    print(solution_3108(n=5, edges=[[0, 1, 7], [1, 3, 7], [1, 2, 1]], query=[[0, 3], [3, 4]]))
    print(solution_3108(n=3, edges=[[0, 2, 7], [0, 1, 15], [1, 2, 6], [1, 2, 1]], query=[[1, 2]]))


def testcase_3108_2():
    print(solution_3108_2(n=5, edges=[[0, 1, 7], [1, 3, 7], [1, 2, 1]], query=[[0, 3], [3, 4]]))
    print(solution_3108_2(n=3, edges=[[0, 2, 7], [0, 1, 15], [1, 2, 6], [1, 2, 1]], query=[[1, 2]]))


def testcase_2009():
    print(solution_2009(nums=[4, 2, 5, 3]))
    print(solution_2009(nums=[1, 2, 3, 5, 6]))
    print(solution_2009(nums=[1, 10, 100, 1000]))


def testcase_405():
    print(solution_405(26))
    print(solution_405(-1))


def testcase_2529():
    print(solution_2529([-2, -1, -1, 1, 2, 3]))
    print(solution_2529([-3, -2, -1, 0, 0, 1, 2]))
    print(solution_2529([5, 20, 66, 1314]))


def testcase_59():
    print(solution_59(3))
    print(solution_59(1))
    print(solution_59(5))


def testcase_1702():
    print(solution_1702("000110"))
    print(solution_1702("01"))
    print(solution_1702("1100"))


def testcase_1702_2():
    print(solution_1702_2("000110"))
    print(solution_1702_2("01"))
    print(solution_1702_2("1100"))


def testcase_1766():
    print(solution_1766(nums=[2, 3, 3, 2], edges=[[0, 1], [1, 2], [1, 3]]))
    print(solution_1766(nums=[5, 6, 10, 2, 3, 6, 15], edges=[[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]))


def testcase_409():
    print(solution_409("abccccdd"))
    print(solution_409("a"))
    print(solution_409("aaaaaccc"))


def testcase_707():
    o = MyLinkedList()
    o.addAtHead(1)
    o.addAtTail(3)
    o.addAtIndex(1, 2)
    print(o.get(1))
    o.deleteAtIndex(1)
    print(o.get(1))

    o = MyLinkedList()
    o.addAtIndex(0, 10)
    o.addAtIndex(0, 20)
    o.addAtIndex(1, 30)
    o.get(0)
