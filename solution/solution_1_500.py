import collections
import math
import os

from .data_struct import *
from .method import *
from .mysql_connecter import *
from collections import defaultdict


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
    stack = Stack(100000)
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


def solution_21(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    if list1 is None:
        return list2
    if list2 is None:
        return list1
    if list1.val < list2.val:
        temp = ListNode(list1.val)
        list1 = list1.next
    else:
        temp = ListNode(list2.val)
        list2 = list2.next
    ret = temp
    while list1 is not None and list2 is not None:
        if list1.val < list2.val:
            temp.next = ListNode(list1.val)
            temp = temp.next
            list1 = list1.next
        else:
            temp.next = ListNode(list2.val)
            temp = temp.next
            list2 = list2.next
    while list1 is not None:
        temp.next = ListNode(list1.val)
        temp = temp.next
        list1 = list1.next
    while list2 is not None:
        temp.next = ListNode(list2.val)
        temp = temp.next
        list2 = list2.next
    return ret


def solution_26(nums: List[int]) -> int:
    if nums is None:
        return 0
    if len(nums) < 2:
        return len(nums)
    p = 1
    temp = nums[0]
    for i in range(1, len(nums)):
        if nums[i] == temp:
            continue
        else:
            temp = nums[i]
            nums[p] = nums[i]
            p += 1
    return p


def solution_27(nums: List[int], val: int) -> int:
    if nums is None:
        return 0
    p = 0
    for i in range(len(nums)):
        if nums[i] != val:
            nums[p] = nums[i]
            p += 1
    return p


def solution_28(haystack: str, needle: str) -> int:
    # 朴素字符串匹配算法
    n = len(haystack)
    m = len(needle)
    for i in range(n - m + 1):
        for j in range(m):
            if haystack[i + j] != needle[j]:
                break
        else:
            return i
    return -1


def solution_28_1(haystack: str, needle: str) -> int:
    # KMP算法

    def compute_next(pattern: str):
        """
        计算next数组
        :param pattern: 模式字符串
        :return: next数组
        """
        pm = len(pattern)  # pattern长度
        ret = [0] * pm  # next数组
        ret[0] = -1  # 默认next[0] = -1
        j = -1  # j为当前匹配的位置
        for k in range(1, pm):
            # 当前匹配失败时,回溯到上一个匹配位置
            # 持续迭代直到匹配成功或者回溯到-1
            while j > -1 and pattern[j + 1] != pattern[k]:
                j = ret[j]
            # 匹配成功时,更新j
            if pattern[j + 1] == pattern[k]:
                j += 1
            ret[k] = j
        return ret

    n = len(haystack)
    m = len(needle)
    pat_next = compute_next(needle)
    q = -1  # 当前匹配的位置
    for i in range(n):
        # 当前匹配失败时,回溯到上一个匹配位置
        # 持续迭代直到匹配成功或者回溯到-1
        while q > -1 and needle[q + 1] != haystack[i]:
            q = pat_next[q]
        # 匹配成功时,更新q
        if needle[q + 1] == haystack[i]:
            q += 1
        # 匹配成功时,返回匹配位置
        if q == m - 1:
            return i - m + 1
    # 匹配失败时,返回-1
    return -1


def solution_35(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    if target < nums[left]:
        return 0
    elif target > nums[right]:
        return len(nums)
    while left < right:
        mid = (left + right) // 2
        if target == nums[mid]:
            return mid
        elif target > nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
    if target <= nums[left]:
        return left
    else:
        return left + 1


def solution_35_2(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if target > nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return left


def solution_58(s: str) -> int:
    p = -1
    q = 0
    k = len(s) - 1
    while s[k] == " ":
        k = k - 1
    while q <= k:
        if s[q] == " ":
            p = q
        q = q + 1
    return q - p - 1


def solution_58_2(s: str) -> int:
    k = len(s) - 1
    while s[k] == " ":
        k = k - 1
    q = k
    while q >= 0:
        if s[q] == " ":
            return k - q
        q = q - 1
    return k - q  # 前端没有空格


def solution_66(digits: List[int]) -> List[int]:
    n = len(digits) - 1
    c = 1  # 进位
    while n >= 0:
        digits[n] = digits[n] + c
        if digits[n] == 10:
            c = 1
            digits[n] = 0
        else:
            return digits
        n = n - 1
    else:
        return [1, *digits]


def solution_67(a: str, b: str) -> str:
    # 使用0补齐短的字符串
    if len(a) < len(b):
        a = '0' * (len(b) - len(a)) + a
    else:
        b = '0' * (len(a) - len(b)) + b
    f_dict = {
        "1": 1,
        "0": 0
    }
    c = 0
    p = len(a) - 1
    ret = list(range(p + 1))
    while p >= 0:
        ret[p] = f_dict[a[p]] + f_dict[b[p]] + c
        if ret[p] >= 2:
            ret[p] = ret[p] - 2
            c = 1
        else:
            c = 0
        p = p - 1
    if c == 1:
        return ''.join(str(num) for num in [1, *ret])
    else:
        return ''.join(str(num) for num in ret)


def solution_69(x: int) -> int:
    # 牛顿迭代
    # x^2 - A = 0
    # 求x
    # 迭代公式 x(n+1) = x(n) - f(x(n))/f'(x(n))
    # 对于本题目 f(x(n)) = x(n)^2 -A, f'(x(n)) = 2x(n)
    x0 = x
    x1 = x0 - (x0 * x0 - x) / (2 * x0)
    while x0 - x1 > 10e-6 or x1 - x0 > 10e-6:
        x0 = x1
        x1 = x0 - (x0 * x0 - x) / (2 * x0)
    return int(x1)


def solution_70(n: int) -> int:
    # 递归 爆栈
    # if n == 0:
    #     return 0
    # if n == 1:
    #     return 1
    # if n == 2:
    #     return 2
    # return solution_70(n - 2) + solution_70(n - 1)
    # 数组存放已求解的值
    if n < 3:
        return n
    ret = list(range(n + 1))
    ret[0] = 0
    ret[1] = 1
    ret[2] = 2
    for i in range(3, n + 1):
        ret[i] = ret[i - 1] + ret[i - 2]
    return ret[n]


def solution_70_2(n: int) -> int:
    """
    矩阵快速幂
    f(x+2) = f(x+1) + f(x)
    -> |f(x+2)| = |1 1| * |f(x+1)|
       |f(x+1)|   |1 0|   |f(x)  |
    -> |f(x+2)| = |1 1| * |1 1| * |f(x)  | = |1 1|^n+1 * |f(1)  |
       |f(x+1)|   |1 0|   |1 0|   |f(x-1)|   |1 0|       |f(0)|
      |1 1| n次幂的求解
      |1 0|
    """

    # 本算法对2*2矩阵的乘法进行特化
    def mul_simple(x, y):
        # x = [..,..,..,..]
        return [x[0] * y[0] + x[1] * y[2],
                x[0] * y[1] + x[1] * y[3],
                x[2] * y[0] + x[3] * y[2],
                x[2] * y[1] + x[3] * y[3]]

    f = [1, 1, 1, 0]
    ans = [1, 0, 0, 1]  # 单位阵
    n = n - 1
    while n > 0:  # 指数不为0
        if n & 1:  # 二进制当前为1,说明结果需要乘以当前底数的2的若干次方
            ans = mul_simple(f, ans)
        f = mul_simple(f, f)  # 计算下一个底数的2的若干次方
        n = n >> 1
    return ans[0] + ans[1]


def solution_412(n: int) -> List[str]:
    answers = list(range(n + 1))
    p = 3
    q = 5
    for i in range(1, n + 1):
        if not (i == p or i == q):
            answers[i] = str(i)
            continue
        answers[i] = ''
        if i == p:
            answers[i] += 'Fizz'
            p = p + 3
        if i == q:
            answers[i] += 'Buzz'
            q = q + 5
    return answers[1:]


def solution_383(ransomNote: str, magazine: str) -> bool:
    f_dict = dict()
    for i in range(len(magazine)):
        temp = magazine[i]
        if temp not in f_dict:
            f_dict[temp] = 1
        else:
            f_dict[temp] += 1
    for i in range(len(ransomNote)):
        temp = ransomNote[i]
        if temp not in f_dict or not f_dict[temp] > 0:
            return False
        else:
            f_dict[temp] -= 1
    return True


def solution_83(head: Optional[ListNode]) -> Optional[ListNode]:
    if head is None:
        return None
    p = head
    while p is not None:
        q = p.next
        if q is None:
            return head
        if q.val == p.val:
            p.next = q.next
            continue
        p = p.next
    return head


def solution_88(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    p = m - 1
    q = n - 1
    k = m + n - 1
    while q >= 0 and p >= 0:
        if nums2[q] <= nums1[p]:
            nums1[k] = nums1[p]
            p -= 1
        else:
            nums1[k] = nums2[q]
            q -= 1
        k -= 1
    while q >= 0:
        nums1[k] = nums2[q]
        k -= 1
        q -= 1
    while p >= 0:
        nums1[k] = nums1[p]
        k -= 1
        p -= 1


def solution_94(root: Optional[TreeNode]) -> List[int]:
    # 先左后中再右
    if root is None:
        return []
    if root.left is None and root.right is None:
        return [root.val]
    # 递归写法
    # return [*solution_94(root.left), root.val, *solution_94(root.right)]
    # 迭代写法 栈
    # s = []
    # p = root
    # res = []
    # while p is not None:
    #     s.append(p)
    #     p = p.left
    # while not len(s) == 0:
    #     top = s.pop()
    #     res.append(top.val)
    #     if top.right is not None:
    #         p = top.right
    #         while p is not None:
    #             s.append(p)
    #             p = p.left
    # return res
    # 更简洁的写法
    s = []
    res = []
    while root is not None or len(s) != 0:
        while root is not None:
            s.append(root)
            root = root.left
        root = s.pop()
        res.append(root.val)
        root = root.right
    return res


def solution_94_2(root: Optional[TreeNode]) -> List[int]:
    # morris
    # 每个节点会被访问两次，但是省去了栈
    res = []
    while root is not None:
        if root.left is not None:
            predecessor = root.left
            while predecessor.right is not None and predecessor.right is not root:
                predecessor = predecessor.right
            if predecessor.right is None:
                predecessor.right = root
                root = root.left
            else:  # 此处为遍历取值的情况，此时左子树遍历完，取root值遍历右子树
                res.append(root.val)
                predecessor.right = None
                root = root.right
        else:
            res.append(root.val)
            root = root.right
    return res


def solution_466(s1: str, n1: int, s2: str, n2: int) -> int:
    if n1 == 0:
        return 0
    s1cnt, s2cnt, index = 0, 0, 0
    # recall 是我们用来找循环节的变量，它是一个哈希映射
    # 我们如何找循环节？假设我们遍历了 s1cnt 个 s1，此时匹配到了第 s2cnt 个 s2 中的第 index 个字符
    # 如果我们之前遍历了 s1cnt' 个 s1 时，匹配到的是第 s2cnt' 个 s2 中同样的第 index 个字符，那么就有循环节了
    # 注意:
    # 在不同的s1末尾出现同一个s2的index,说明从此处开始匹配和最开始开始匹配的过程将完全一致
    # 我们用 (s1cnt', s2cnt', index) 和 (s1cnt, s2cnt, index) 表示两次包含相同 index 的匹配结果
    # 那么哈希映射中的键就是 index，值就是 (s1cnt', s2cnt') 这个二元组
    # 循环节就是；
    #    - 前 s1cnt' 个 s1 包含了 s2cnt' 个 s2
    #    - 以后的每 (s1cnt - s1cnt') 个 s1 包含了 (s2cnt - s2cnt') 个 s2
    # 那么还会剩下 (n1 - s1cnt') % (s1cnt - s1cnt') 个 s1, 我们对这些与 s2 进行暴力匹配
    # 注意 s2 要从第 index 个字符开始匹配
    recall = {}
    while True:
        for ch in s1:
            if ch == s2[index]:
                index += 1
                if index == len(s2):
                    s2cnt, index = s2cnt + 1, 0  # 匹配完了1个s2
        # 遍历完了一个s1,考察此时index的位置
        s1cnt += 1
        if s1cnt == n1:
            return s2cnt // n2  # n1用完没找到循环节
        if index in recall:
            # 前s1cnt_prime个s1包含了s2cnt_prime个s2,并拼到了s2的index位置
            s1cnt_prime, s2cnt_prime = recall[index]
            pre_loop = (s1cnt_prime, s2cnt_prime)
            in_loop = (s1cnt - s1cnt_prime, s2cnt - s2cnt_prime)
            break
        else:
            recall[index] = (s1cnt, s2cnt)
    # ans保存循环节中匹配的s2的数量
    ans = pre_loop[1] + (n1 - pre_loop[0]) // in_loop[0] * in_loop[1]
    rest = (n1 - pre_loop[0]) % in_loop[0]
    for i in range(rest):
        for ch in s1:
            if ch == s2[index]:
                index += 1
                if index == len(s2):
                    ans, index = ans + 1, 0  # 匹配完了1个s2
    return ans // n2


def solution_466_2(s1: str, n1: int, s2: str, n2: int) -> int:
    # 我们预处理出以字符串 s2 的每个位置 i 开始匹配一个完整的 s1 后，下一个位置 j 以及经过了多少个 s2，即 d[i]=(cnt,j)，
    # 其中 cnt 表示匹配了多少个 s2，而 j 表示字符串 s2 的下一个位置。
    # 接下来，我们初始化 j=0，然后循环 n1 次，每一次将 d[j][0] 加到答案中，然后更新 j=d[j][1]
    # 最后得到的答案就是 n1 个 s1 所能匹配的 s2 的个数，除以 n2 即可得到答案
    n = len(s2)
    d = {}
    for i in range(n):
        cnt = 0
        j = i
        for c in s1:
            if c == s2[j]:
                j += 1
            if j == n:
                cnt += 1
                j = 0
        d[i] = (cnt, j)
    ans = 0
    j = 0
    for _ in range(n1):
        cnt, j = d[j]
        ans += cnt
    return ans // n2


def solution_100(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    # 递归写法
    # DFS
    # if p is None and q is None:
    #     return True
    # elif p is None and q is not None:
    #     return False
    # elif q is None and p is not None:
    #     return False
    # return p.val == q.val and solution_100(p.left, q.left) and solution_100(p.right, q.right)
    # 迭代写法
    # BFS
    bfs_q = queue.Queue()
    if p is None and q is None:
        return True
    bfs_q.put(p)
    bfs_q.put(q)
    while not bfs_q.empty():
        t1 = bfs_q.get()
        t2 = bfs_q.get()
        if t1 is None and t2 is None:
            continue
        if t1 is None or t2 is None or t1.val != t2.val:
            return False
        bfs_q.put(t1.left)
        bfs_q.put(t2.left)
        bfs_q.put(t1.right)
        bfs_q.put(t2.right)
    return True


def solution_101(root: Optional[TreeNode]) -> bool:
    if root is None:
        return True
    bfs_q = queue.Queue()
    bfs_q.put(root.left)
    bfs_q.put(root.right)
    while not bfs_q.empty():
        t1 = bfs_q.get()
        t2 = bfs_q.get()
        if t1 is None and t2 is None:
            continue
        if t1 is None or t2 is None or t1.val != t2.val:
            return False
        bfs_q.put(t1.left)
        bfs_q.put(t2.right)
        bfs_q.put(t1.right)
        bfs_q.put(t2.left)
    return True


def solution_104(root: Optional[TreeNode]) -> int:
    # if root is None:
    #     return 0
    # bfs_q = queue.Queue()
    # bfs_q.put(root)
    # label = TreeNode(-101)
    # bfs_q.put(label)
    # count = 0
    # while not bfs_q.empty():
    #     t = bfs_q.get()
    #     if t.val == -101:
    #         count += 1
    #         if bfs_q.empty():
    #             return count
    #         bfs_q.put(label)
    #         continue
    #     if t.left is not None:
    #         bfs_q.put(t.left)
    #     if t.right is not None:
    #         bfs_q.put(t.right)
    # 上面的有点麻烦了
    # 每一层用一个队列，用完丢掉就可以了
    if not root:
        return 0
    my_queue, res = [root], 0
    while my_queue:
        tmp = []
        for node in my_queue:
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
        my_queue = tmp
        res += 1
    return res


def solution_108(nums: List[int]) -> Optional[TreeNode]:
    # bfs_q = queue.Queue()
    # n = len(nums)
    # mid = n // 2
    # root = TreeNode(nums[mid])
    # bfs_q.put((0, mid, root))
    # bfs_q.put((mid + 1, n, root))
    # while not bfs_q.empty():
    #     llow, lhigh, p = bfs_q.get()
    #     if llow < lhigh:
    #         lmid = (lhigh - llow) // 2 + llow
    #         p.left = TreeNode(nums[lmid])
    #         bfs_q.put((llow, lmid, p.left))
    #         bfs_q.put((lmid + 1, lhigh, p.left))
    #     rlow, rhigh, p = bfs_q.get()
    #     if rlow < rhigh:
    #         rmid = (rhigh - rlow) // 2 + rlow
    #         p.right = TreeNode(nums[rmid])
    #         bfs_q.put((rlow, rmid, p.right))
    #         bfs_q.put((rmid + 1, rhigh, p.right))
    # return root
    bfs_q = []
    n = len(nums)
    mid = n // 2
    root = TreeNode(nums[mid])
    bfs_q.append((0, mid, root))
    bfs_q.append((mid + 1, n, root))
    while bfs_q:
        tmp = []
        i = 0
        while i < len(bfs_q):
            llow, lhigh, p = bfs_q[i]
            if llow < lhigh:
                lmid = (lhigh - llow) // 2 + llow
                p.left = TreeNode(nums[lmid])
                tmp.append((llow, lmid, p.left))
                tmp.append((lmid + 1, lhigh, p.left))
            rlow, rhigh, p = bfs_q[i + 1]
            if rlow < rhigh:
                rmid = (rhigh - rlow) // 2 + rlow
                p.right = TreeNode(nums[rmid])
                tmp.append((rlow, rmid, p.right))
                tmp.append((rmid + 1, rhigh, p.right))
            i += 2
        bfs_q = tmp
    return root


def solution_110(root: Optional[TreeNode]) -> bool:
    # 递归
    def helper(p):
        if p is None:
            return 0
        left_height = helper(p.left)
        right_height = helper(p.right)
        # 用-1标记不平衡
        if left_height == -1 or right_height == -1 or abs(left_height - right_height) > 1:
            return -1
        else:
            return max(left_height, right_height) + 1

    return helper(root) >= 0


def solution_111(root: Optional[TreeNode]) -> int:
    # 递归
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    elif root.left is None:
        return solution_111(root.right) + 1
    elif root.right is None:
        return solution_111(root.left) + 1
    else:
        return min(solution_111(root.left), solution_111(root.right)) + 1


def solution_111_2(root: Optional[TreeNode]) -> int:
    # bfs
    if not root:
        return 0
    que = collections.deque([(root, 1)])
    while que:
        node, depth = que.popleft()
        if not node.left and not node.right:
            return depth
        if node.left:
            que.append((node.left, depth + 1))
        if node.right:
            que.append((node.right, depth + 1))
    return 0


def solution_112(root: Optional[TreeNode], targetSum: int) -> bool:
    # 递归
    # if not root:
    #     return False
    # if not root.left or not root.right:
    #     return targetSum == root.val
    # return solution_112(root.left, targetSum - root.val) or solution_112(root.right, targetSum - root.val)
    if not root:
        return False
    stack = [(root, False)]
    total = 0
    while stack:
        node, flag = stack[-1]
        if not node:
            stack.pop()
            continue
        if flag:
            total = total - node.val
            stack.pop()
            continue
        if not node.left and not node.right and total + node.val == targetSum:
            return True
        total += node.val
        stack[-1] = (node, True)
        if node.left:
            stack.append((node.left, False))
        if node.right:
            stack.append((node.right, False))
    return False


def solution_118(numRows: int) -> List[List[int]]:
    if numRows == 1:
        return [[1]]
    if numRows == 2:
        return [[1], [1, 1]]
    res = [[1], [1, 1]]
    for i in range(2, numRows):
        tmp = [0] * (i + 1)
        last = res[i - 1]
        for j in range(1, i):
            tmp[j] = last[j - 1] + last[j]
        tmp[0], tmp[-1] = 1, 1
        res.append(tmp)
    return res


def solution_119(rowIndex: int) -> List[int]:
    res = [1] * (rowIndex + 1)
    # 正向计算
    # 由于需要j-1处的数值，所以需要保留副本
    # for i in range(1, rowIndex + 1):
    #     last = list(res)
    #     for j in range(1, i):
    #         res[j] = last[j - 1] + last[j]
    for i in range(1, rowIndex + 1):
        for j in range(i - 1, 0, -1):
            res[j] = res[j - 1] + res[j]
    return res


def solution_121(prices: List[int]) -> int:
    # 超出内存限制
    # # dp
    # n = len(prices)
    # profit = []
    # for i in range(n):
    #     profit.append([0] * n)
    # # 初始化
    # for i in range(n - 1):
    #     profit[i][i + 1] = prices[i + 1] - prices[i]
    # for i in range(n - 2, 0, -1):
    #     for j in range(i):
    #         profit[j][j + n - i] = profit[j][j + n - i - 1] + profit[j + n - i - 1][j + n - i]
    # return max(max(x) for x in profit)
    # 先找卖出日前的最小值，记录最大利润值
    inf = int(1e9)
    min_price = inf
    max_profit = 0
    for price in prices:
        max_profit = max(price - min_price, max_profit)  # 保证非全局最小点取到的最优值也能被记录并返回，如3，1000，1，5
        min_price = min(price, min_price)
    return max_profit


def solution_125(s: str) -> bool:
    # tmp = ""
    # for i in range(len(s)):
    #     num = ord(s[i])
    #     if 65 <= num <= 90:
    #         tmp += chr(num + 32)
    #     elif 97 <= num <= 122:
    #         tmp += s[i]
    #     elif 48 <= num <= 57:
    #         tmp += s[i]
    # for i in range(len(tmp) // 2):
    #     if tmp[i] != tmp[len(tmp) - 1 - i]:
    #         return False
    # return True
    # 不做预处理
    left = 0
    right = len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if left < right:
            if s[left].lower() != s[right].lower():
                return False
            left, right = left + 1, right - 1
    return True


def solution_136(nums: List[int]) -> int:
    # a ^ b ^ c <= > a ^ c ^ b
    # 0 ^ n = > n
    # 相同的数异或为0: n ^ n = > 0
    res = 0
    for num in nums:
        res ^= num
    return res
    # 两个数奇数次
    # eor, eor1 = 0, 0
    # for num in nums:
    #     eor ^= num
    # right = eor & (-eor) # 两个数肯定不相等，因此至少有1位为1，用这个1做分组
    # for num in nums:
    #     if right & num:
    #         eor1 ^= num
    # return eor1, eor ^ eor1 # eor是两个数异或的结果，再异或一次另一个值就出来了


def solution_447(points: List[List[int]]) -> int:
    def distance(x1, x2):
        return (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2

    count = 0
    for point in points:
        f_dict = {}
        for point1 in points:
            dis = distance(point, point1)
            f_dict[dis] = f_dict.get(dis, 0) + 1
        for x in f_dict:
            if f_dict[x] >= 2:
                count += math.perm(f_dict[x], 2)
    return count


def solution_141(head: Optional[ListNode]) -> bool:
    if not head or not head.next:
        return False
    p, q, = head.next.next, head.next
    while p and q:
        if p == q:
            return True
        if not p.next or not p.next.next:
            return False
        p = p.next.next
        q = q.next
    return False


def solution_144(root: Optional[TreeNode]) -> List[int]:
    # 递归
    # def helper(p, res):
    #     if not p:
    #         return
    #     res.append(p.val)
    #     helper(p.left, res)
    #     helper(p.right, res)
    #
    # result = []
    # helper(root, result)
    # return result
    if not root:
        return []
    result = []
    # stack = [(root, False)]
    # while stack:
    #     p, flag = stack[-1]
    #     if not flag:
    #         result.append(p.val)
    #         stack[-1] = p, True
    #         if p.right:
    #             stack.append((p.right, False))
    #         if p.left:
    #             stack.append((p.left, False))
    #     else:
    #         stack.pop()
    # return result
    stack = []
    p = root
    while stack or p:
        while p:
            result.append(p.val)
            stack.append(p)
            p = p.left
        p = stack.pop()
        p = p.right
    return result


def solution_144_2(root: Optional[TreeNode]) -> List[int]:
    # morris
    res = []
    p = root
    while p:
        if p.left:
            predecessor = p.left
            while predecessor.right and not predecessor.right == p:
                predecessor = predecessor.right
            if predecessor.right == p:
                predecessor.right = None
                p = p.right
            else:
                predecessor.right = p
                res.append(p.val)
                p = p.left
        else:
            res.append(p.val)
            p = p.right
    return res


def solution_145(root: Optional[TreeNode]) -> List[int]:
    # def helper(p, res):
    #     if not p:
    #         return
    #     helper(p.left, res)
    #     helper(p.right, res)
    #     res.append(p.val)

    res = []

    # helper(root, result)
    # return result
    # 迭代
    # stack = []
    # p = root
    # prev = None
    # while stack or p:
    #     while p:
    #         stack.append(p)
    #         p = p.left
    #     p = stack.pop()
    #     if not p.right or p.right == prev:
    #         result.append(p.val)
    #         prev = p  # 说明p的右子树遍历完了,该输出自己的值了
    #         p = None
    #     else:
    #         stack.append(p)
    #         p = p.right
    # return result
    # morris
    # 该函数每次会拿出树的最右侧一排数值
    # 而后序遍历时会从最左侧开始,一次次地拿最右侧的一排数值
    def addPath(node: TreeNode):
        count = 0
        while node:
            count += 1
            res.append(node.val)
            node = node.right
        # 前后反转
        i, j = len(res) - count, len(res) - 1
        while i < j:
            res[i], res[j] = res[j], res[i]
            i += 1
            j -= 1

    if not root:
        return list()

    p = root
    while p:
        if p.left:
            predecessor = p.left
            while predecessor.right and predecessor.right != p:
                predecessor = predecessor.right
            if predecessor.right == p:
                predecessor.right = None
                addPath(p.left)
                p = p.right
            else:
                predecessor.right = p
                p = p.left
        else:
            p = p.right
    addPath(root)
    return res


def solution_208():
    # 该题需要建立trie树,详见data_struct.py
    pass


def solution_160(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """
    移除较长链表的开头部分,两个指针共同前进
    """
    p, q, m, n = headA, headB, 0, 0
    while p:
        m += 1
        p = p.next
    while q:
        n += 1
        q = q.next
    p, q = headA, headB
    if m >= n:
        tmp = m - n
        for _ in range(tmp):
            p = p.next
    else:
        tmp = n - m
        for _ in range(tmp):
            q = q.next
    while p and q:
        if p == q:
            return p
        p = p.next
        q = q.next
    return None


def solution_160_2(headA: ListNode, headB: ListNode) -> Optional[ListNode]:
    """
    考虑两种情况，第一种情况是两个链表相交，第二种情况是两个链表不相交。
    情况一：两个链表相交
    链表 headA 和 headB 的长度分别是 m 和 n。
    假设链表 headA 的不相交部分有 a个节点，链表 headB 的不相交部分有 b 个节点，两个链表相交的部分有 c 个节点，
    则有 a+c=m，b+c=n。
    如果 a=b，则两个指针会同时到达两个链表相交的节点，此时返回相交的节点；
    如果 a≠b，则指针 pApA 会遍历完链表 headA，指针 pB 会遍历完链表 headB，两个指针不会同时到达链表的尾节点，
    然后指针 pA 移到链表 headB 的头节点，指针 pB 移到链表 headA 的头节点，然后两个指针继续移动，
    在指针 pA 移动了 a+c+b 次、指针 pB移动了 b+c+a 次之后，两个指针会同时到达两个链表相交的节点，该节点也是两个指针第一次同时指向的节点，此时返回相交的节点。
    情况二：两个链表不相交
    链表 headA 和 headB 的长度分别是 m 和 n。考虑当 m=n 和 m≠n时，两个指针分别会如何移动：
    如果 m=n 则两个指针会同时到达两个链表的尾节点，然后同时变成空值 null，此时返回 null；
    如果 m≠n 则由于两个链表没有公共节点，两个指针也不会同时到达两个链表的尾节点，因此两个指针都会遍历完两个链表，在指针 pA 移动了 m+n 次、指针 pB 移动了 n+m 次之后，两个指针会同时变成空值 null，此时返回 null
    """
    p1, p2 = headA, headB
    while p1 != p2:
        p1 = p1.next if p1 is not None else headB
        p2 = p2.next if p2 is not None else headA
    return p1


def solution_168(columnNumber: int) -> str:
    s = []
    while columnNumber > 0:
        columnNumber -= 1
        remainder = columnNumber % 26
        columnNumber = columnNumber // 26
        s.append(chr(remainder + ord("A")))
    return "".join(s[::-1])


def solution_171(columnTitle: str) -> int:
    res = 0
    for x in columnTitle:
        res *= 26
        res += ord(x) - ord("A") + 1
    return res


def solution_169(nums: List[int]) -> int:
    # 简单思路哈希表即可
    # Boyer-Moore 多数元素最后一定比其他元素多
    count, candidate = 0, None
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if candidate == num else -1
    return candidate


def solution_175():
    # 为什么会有SQL题目?
    # PostgreSql
    # select firstName,lastName,city,state from  Person left join  Address on Person.personId = Address.personId
    sql = "select firstName,lastName,city,state from  Person left join  Address on Person.personId = Address.personId"
    return execute(sql)


def solution_181():
    # PostgreSql
    # select P1.name from Employee as P1 join Employee as P2 on P1.managerId = P2.id where P1.salary > P2.salary
    sql = "select P1.name from Employee P1 join Employee P2 on P1.managerId = P2.id where P1.salary > P2.salary"
    return execute(sql)


def solution_182() -> Any:
    sql = "select email from (select count(id) as num,email from Person group by email ) as ne where num > 1"
    return execute(sql)


def solution_183() -> Any:
    sql = "select name as customers from Customers as c left join Orders as o on c.id = o.customerId where o.id is null"
    return execute(sql)


def solution_190(n: int) -> int:
    # res = 0
    # for i in range(0, 31):
    #     res += (n >> i) & 1
    #     res <<= 1
    # res += (n >> 31) & 1
    # return res
    res = 0
    i = 0
    while i < 32 and n > 0:
        res = res | ((n & 1) << (31 - i))
        n >>= 1
        i += 1
    return res


def solution_190_2(n: int) -> int:
    M1 = 0x55555555  # 01010101010101010101010101010101
    M2 = 0x33333333  # 00110011001100110011001100110011
    M4 = 0x0f0f0f0f  # 00001111000011110000111100001111
    M8 = 0x00ff00ff  # 00000000111111110000000011111111

    def trunc(tmp):
        return tmp & 0xffffffff

    n = trunc(n >> 1) & M1 | trunc((n & M1) << 1)
    n = trunc(n >> 2) & M2 | trunc((n & M2) << 2)
    n = trunc(n >> 4) & M4 | trunc((n & M4) << 4)
    n = trunc(n >> 8) & M8 | trunc((n & M8) << 8)
    return trunc(n >> 16) | trunc(n << 16)


def solution_191(n: int) -> int:
    M1 = 0x55555555  # 01010101010101010101010101010101
    M2 = 0x33333333  # 00110011001100110011001100110011
    M4 = 0x0f0f0f0f  # 00001111000011110000111100001111
    M8 = 0x00ff00ff  # 00000000111111110000000011111111
    M16 = 0x0000ffff  # 00000000000000001111111111111111
    n = (n & M1) + ((n >> 1) & M1)
    n = (n & M2) + ((n >> 2) & M2)
    n = (n & M4) + ((n >> 4) & M4)
    n = (n & M8) + ((n >> 8) & M8)
    return (n & M16) + (n >> 16 & M16)


def solution_193():
    # 还有shell题目?
    os.system("egrep '(^[0-9]{3}-[0-9]{3}-[0-9]{4}$)|(^\([0-9]{3}\) [0-9]{3}-[0-9]{4}$)' resources/file.txt")
    os.system("sed -n -r  '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/p' resources/file.txt")


def solution_195():
    os.system("sed -n '10 p' resources/file.txt")  # 打印第10行
    os.system("tail -n +10 resources/file.txt | head -1")  # 从第十行开始读所有行,管道传给head读第一行
    os.system("awk 'NR == 10' resources/file.txt")  # 处理到第10行,打印


def solution_202(n: int) -> bool:
    f = {}
    while True:
        tmp = n
        n = 0
        while tmp:
            n, tmp = n + (tmp % 10) ** 2, tmp // 10
        if n == 1:
            return True
        if n in f:
            return False
        f[n] = 1


def solution_202_2(n: int) -> bool:
    # 任何数最终都会回到1到243之间
    # 999 -> 243
    # 9999 -> 324
    # 9999999999999 -> 1053
    # 位数会不断减少直到回到三位数
    # 存在循环，可以快慢指针
    def get_next(tmp):
        total = 0
        while tmp:
            total, tmp = total + (tmp % 10) ** 2, tmp // 10
        return total

    s = n
    f = get_next(n)
    while f != 1 and s != f:
        s = get_next(s)
        f = get_next(get_next(f))
    return f == 1


def solution_203(head: Optional[ListNode], val: int) -> Optional[ListNode]:
    v_head = ListNode(-1)
    v_head.next = head
    p, q = v_head.next, v_head
    while p:
        if p.val == val:
            q.next = p.next
            p = p.next
        else:
            p = p.next
            q = q.next
    return v_head.next


def solution_205(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    f = {}
    my_set = set()
    for i in range(len(s)):
        if s[i] not in f:
            if t[i] in my_set:
                return False
            my_set.add(t[i])
            f[s[i]] = t[i]
        elif f[s[i]] != t[i]:
            return False
    return True


def solution_290(pattern: str, s: str) -> bool:
    words = s.split(" ")
    if len(pattern) != len(words):
        return False
    f_words_to_pattern = {}
    s_pattern = set()
    for i in range(len(words)):
        if words[i] not in f_words_to_pattern:
            f_words_to_pattern[words[i]] = pattern[i]
            if pattern[i] in s_pattern:
                return False
            s_pattern.add(pattern[i])
        elif f_words_to_pattern[words[i]] != pattern[i]:
            return False
    return True


def solution_2(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    c = 0
    res = ListNode(-1)
    p = res
    while l1 and l2:
        tmp = c + l1.val + l2.val
        if tmp >= 10:
            tmp, c = tmp - 10, 1
        else:
            c = 0
        p.next = ListNode(tmp)
        p = p.next
        l1 = l1.next
        l2 = l2.next
    while l1:
        tmp = c + l1.val
        if tmp >= 10:
            tmp, c = tmp - 10, 1
        else:
            c = 0
        p.next = ListNode(tmp)
        p = p.next
        l1 = l1.next
    while l2:
        tmp = c + l2.val
        if tmp >= 10:
            tmp, c = tmp - 10, 1
        else:
            c = 0
        p.next = ListNode(tmp)
        p = p.next
        l2 = l2.next
    if c:
        p.next = ListNode(1)
    return res.next


def solution_3(s: str) -> int:
    max_count, count = 0, 0
    last = -1
    f = {}
    for i in range(len(s)):
        if s[i] in f:
            count = i - last - 1
            last = f[s[i]]
            f.clear()
            for j in range(last + 1, i):
                f[s[j]] = j
            if count > max_count:
                max_count = count
        f[s[i]] = i
    count = len(s) - last - 1
    return max(max_count, count)


def solution_3_2(s: str) -> int:
    # 理解滑动窗口
    my_set = set()
    n = len(s)
    rk, ans = 0, 0
    for i in range(n):
        if i != 0:
            my_set.remove(s[i - 1])
        while rk < n and s[rk] not in my_set:
            my_set.add(s[rk])
            rk += 1
        ans = max(ans, rk - i)
    return ans


def solution_3_3(s: str) -> int:
    f = {}
    ans, left = 0, 0
    i = 0
    for i in range(len(s)):
        if s[i] in f and left <= f[s[i]]:
            ans = max(ans, i - left)
            left = f[s[i]] + 1
        f[s[i]] = i
    return max(ans, len(s) - left)


def solution_82(head: Optional[ListNode]) -> Optional[ListNode]:
    v_head = ListNode(-101, head)
    p, q = v_head, v_head.next
    while q and q.next:
        if q.val == q.next.val:
            cur = q.val
            q = q.next.next
            while q and q.val == cur:
                q = q.next
            p.next = q
        else:
            p = q
            q = q.next
    return v_head.next


def solution_196() -> Any:
    sql = ("DELETE from Person where id in (select id from (select p1.id from Person p1 join Person p2 on p1.email = "
           "p2.email and p1.id > p2.id) t)")
    execute(sql)


def solution_197() -> Any:
    sql = ("select w1.id from Weather w1 join Weather w2 on date_sub(w1.recordDate,interval 1 day ) = w2.recordDate "
           "and w1.temperature>w2.temperature")
    return execute(sql)


def solution_206(head: Optional[ListNode]) -> Optional[ListNode]:
    dummy_head = ListNode(-1)
    p = head
    while p:
        q = p.next
        p.next = dummy_head.next
        dummy_head.next = p
        p = q
    return dummy_head.next


def solution_217(nums: List[int]) -> bool:
    my_set = set()
    for num in nums:
        if num in my_set:
            return True
        else:
            my_set.add(num)
    return False


def solution_233(n: int) -> int:
    s = str(n)
    dp = [[-1] * len(s) for _ in range(len(s))]

    def dfs(i: int, j: int, limit: bool) -> int:
        if i == len(s):
            return j
        if not limit and dp[i][j] != -1:
            return dp[i][j]
        res = 0
        up = int(s[i]) if limit else 9
        for d in range(up + 1):
            res += dfs(i + 1, j + (d == 1), limit and d == up)
        if not limit:
            dp[i][j] = res
        return res

    return dfs(0, 0, True)


def solution_219(nums: List[int], k: int) -> bool:
    f = {}
    for i in range(len(nums)):
        if nums[i] in f:
            if i - f[nums[i]] <= k:
                return True
        f[nums[i]] = i
    return False


def solution_219_2(nums: List[int], k: int) -> bool:
    my_set = set()
    for i in range(len(nums)):
        if i > k:
            my_set.remove(nums[i - k - 1])
        if nums[i] in my_set:
            return True
        my_set.add(nums[i])
    return False


def solution_222(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    if not root.left and not root.right:
        return 1
    sl, sr = 0, 0
    if root.left:
        sl = solution_222(root.left)
    if root.right:
        sr = solution_222(root.right)
    return 1 + sl + sr


def solution_222_2(root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    # 二分查找 + 位运算
    def exist(k):
        bits = 1 << (count - 1)
        node = root
        while bits > 0 and node:
            node = node.left if bits & k == 0 else node.right
            bits >>= 1
        return node is not None

    # 遍历最左子树找到高度
    count = 0
    p = root.left
    while p:
        count += 1
        p = p.left
    # 节点个数位于[2**h,2**(h+1)-1]之间
    # 检查k是否存在
    # 位运算
    # 12 -> 1100,第3层 -> 移除最前位 -> 100 -> 右左左
    low = 2 ** count
    high = 2 ** (count + 1) - 1
    while low < high:
        mid = (high - low + 1) // 2 + low
        if exist(mid):
            low = mid
        else:
            high = mid - 1
    return low


def solution_225():
    # 用两个队列实现栈
    # data_struct.StackWithQueue
    ...


def solution_226(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    root.left, root.right = root.right, root.left
    solution_226(root.left)
    solution_226(root.right)
    return root


def solution_228(nums: List[int]) -> List[str]:
    start = 0
    res = []
    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] > 1:
            if start == i - 1:
                res.append(str(nums[start]))
            else:
                res.append(str(nums[start]) + '->' + str(nums[i - 1]))
            start = i
    if start < len(nums):
        res.append(str(nums[start]) + '->' + str(nums[len(nums) - 1]))
    if start == len(nums) - 1:
        res.append(str(nums[start]))
    return res


def solution_410(nums: List[int], k: int) -> int:
    def check(mx: int) -> bool:
        cnt = 1
        s = 0
        for x in nums:
            if s + x <= mx:
                s += x
            else:
                if cnt == k:
                    return False
                cnt += 1
                s = x
        return True

    right = sum(nums)  # 必定满足条件，分成k段，每段的和都会比这个值小
    # 如果分成k段，每段的值都比S/K小，是不可能的，最后的结果一定比这个大
    # 如果最大值比数组里最大的值-1小，也是不可能的
    # 从而确定下界
    left = max(max(nums) - 1, (right - 1) // k)
    # 确定上下界后，二分查找可能的答案，也就是可能的分段数组的最大值
    # 可以前缀和
    # 逐个检查是否比mid大，大则分一段新的
    while left + 1 < right:
        mid = (left + right) // 2
        if check(mid):
            right = mid
        else:
            left = mid
    return right


def solution_410_2(nums: List[int], k: int) -> int:
    # 加了前缀和反而慢了
    # 不知道为何
    s = [nums[0]]
    for i in range(1, len(nums)):
        s.append(nums[i] + s[i - 1])

    def check(mx: int) -> bool:
        cnt = 1
        last = 0
        for j in range(len(nums)):
            if s[j] - last > mx:
                if cnt == k:
                    return False
                else:
                    cnt += 1
                    last = s[j - 1]
        return True

    right = sum(nums)  # 必定满足条件，分成k段，每段的和都会比这个值小
    # 如果分成k段，每段的值都比S/K小，是不可能的，最后的结果一定比这个大
    # 如果最大值比数组里最大的值-1小，也是不可能的
    # 从而确定下界
    left = max(max(nums) - 1, (right - 1) // k)
    # 确定上下界后，二分查找可能的答案，也就是可能的分段数组的最大值
    # 可以前缀和
    # 逐个检查是否比mid大，大则分一段新的
    while left + 1 < right:
        mid = (left + right) // 2
        if check(mid):
            right = mid
        else:
            left = mid
    return right


def solution_410_3(nums: List[int], k: int) -> int:
    # dp[i][j]: 将数组前i个数分割为j段所能得到的最大连续子数组和的最小值
    # 考虑第j段的具体范围，枚举k
    # dp[i][j] = max(dp[k][j-1],sub(k+1,i)),k从0到i-1
    # 最终答案为dp[n][m]
    n = len(nums)
    dp = [[10 ** 18] * (k + 1) for _ in range(n + 1)]
    sub = [0]
    for num in nums:
        sub.append(sub[-1] + num)
    dp[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            for m in range(i):
                dp[i][j] = min(dp[i][j], max(dp[m][j - 1], sub[i] - sub[m]))
    return dp[n][k]


def solution_231(n: int) -> bool:
    if n <= 0:
        return False
    return n & (-n) == n
    # if (n | low_bit) ^ low_bit == 0:
    #     return True
    # else:
    #     return False


def solution_231_2(n: int) -> bool:
    if n <= 0:
        return False
    return n & (n - 1) == 0


def solution_232():
    # 用栈实现队列
    # data_struct.QueueWithStack
    ...


def solution_234(head: Optional[ListNode]) -> bool:
    # 找到中点 + 反转链表
    if not head or not head.next:
        return True
    p, q = head, head
    while p and p.next:
        p = p.next.next
        q = q.next
    if p:
        q = q.next
    dummy_head = ListNode(-1)
    while q:
        q1 = q.next
        q.next = dummy_head.next
        dummy_head.next = q
        q = q1
    p1 = head
    p2 = dummy_head.next
    while p2:
        if p1.val != p2.val:
            return False
        p1 = p1.next
        p2 = p2.next
    return True


def solution_242(s: str, t: str) -> bool:
    f = {}
    for ch in s:
        if ch in f:
            f[ch] += 1
        else:
            f[ch] = 1
    for ch in t:
        if ch not in f:
            return False
        else:
            f[ch] -= 1
            if f[ch] == 0:
                f.pop(ch)
    return len(f) == 0


def solution_275(citations: List[int]) -> int:
    def check(n: int) -> bool:
        if citations[-n] >= n:
            return True
        else:
            return False

    left = 0
    right = len(citations) + 1
    while left + 1 < right:
        mid = (left + right) // 2
        if check(mid):
            left = mid
        else:
            right = mid
    return left


def solution_365(jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
    if jug1Capacity + jug2Capacity < targetCapacity:
        return False
    if jug1Capacity == 0 or jug2Capacity == 0:
        return targetCapacity == 0 or jug1Capacity + jug2Capacity == targetCapacity
    return targetCapacity % gcd_euclid(jug1Capacity, jug2Capacity) == 0


def solution_365_2(jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
    # 保存的状态（1中的水，2中的水）
    stack = [(0, 0)]
    seen = set()
    while stack:
        remain_x, remain_y = stack.pop()
        if remain_x + remain_y == targetCapacity:
            return True
        if (remain_x, remain_y) in seen:
            continue
        seen.add((remain_x, remain_y))
        stack.append((jug1Capacity, remain_y))  # 加满1
        stack.append((remain_x, jug2Capacity))  # 加满2
        stack.append((remain_x, 0))  # 清空2
        stack.append((0, remain_y))  # 清空1
        stack.append(
            (remain_x - min(remain_x, jug2Capacity - remain_y), remain_y + min(remain_x, jug2Capacity - remain_y)))
        stack.append(
            (remain_x + min(remain_y, jug1Capacity - remain_x), remain_y - min(remain_y, jug1Capacity - remain_x)))
    return False


def solution_292(n: int) -> bool:
    return n % 4 != 0


def solution_292_2(n: int) -> bool:
    """
    超时
    """
    if n <= 3:
        return True
    dp = [True] * (n + 1)
    for i in range(4, n + 1):
        dp[i] = (not dp[i - 1]) or (not dp[i - 2]) or (not dp[i - 3])
    return dp[n]


def solution_236(root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
    f = {}
    depth = {}

    def dfs(p, q):
        depth[p] = depth[q] + 1
        f[p] = q
        if p.left:
            f[p.left] = p
            dfs(p.left, p)
        if p.right:
            f[p.right] = p
            dfs(p.right, p)

    depth[root] = 0
    f[root] = None
    if root.left:
        dfs(root.left, root)
    if root.right:
        dfs(root.right, root)

    while p != q:
        if depth[p] > depth[q]:
            p = f[p]
        else:
            q = f[q]
    return p


def solution_236_2(root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
    if root is None or root == p or root == q:
        return root
    left = solution_236_2(root.left, p, q)
    right = solution_236_2(root.right, p, q)
    if left and right:
        return root
    return left or right


def solution_236_3(root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    适合一棵树做多次查询
    预处理过程只需要做一边
    """
    f = {}
    depth = {}
    cnt = 1

    def dfs(p, q):
        nonlocal cnt
        cnt += 1
        depth[p] = depth[q] + 1
        f[p] = q
        if p.left:
            f[p.left] = p
            dfs(p.left, p)
        if p.right:
            f[p.right] = p
            dfs(p.right, p)

    depth[root] = 0
    f[root] = None
    if root.left:
        dfs(root.left, root)
    if root.right:
        dfs(root.right, root)

    m = cnt.bit_length()
    for x in f:
        tmp = f[x]
        f[x] = [None] * m
        f[x][0] = tmp
    for i in range(m - 1):
        for x in f:
            if (tmp_node := f[x][i]) is not None:
                f[x][i + 1] = f[tmp_node][i]  # 更新2倍幂上跳对应的节点

    def get_kth(node, k: int) -> int:
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                node = f[node][i]
                if node is None:
                    break
        return node

    if depth[p] > depth[q]:
        p, q = q, p
    q = get_kth(q, depth[q] - depth[p])  # 使得node2和node1位于同一深度
    if p == q:
        return p
    # 贪心上跳
    for i in range(len(f[q]) - 1, -1, -1):
        p1, p2 = f[p][i], f[q][i]
        if p1 != p2:  # 还能继续往上跳
            p, q = p1, p2
    return f[p][0]


def solution_102(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = [root]
    ans = []
    while q:
        tmp = q
        q = []
        res = []
        for node in tmp:
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        ans.append(res)
    return ans


def solution_103(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = [root]
    ans = []
    reverse = False
    while q:
        tmp = q
        q = []
        res = []
        for node in tmp:
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        if reverse:
            res = res[::-1]
        reverse = not reverse
        ans.append(res)
    return ans


def solution_107(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    q = [root]
    ans = []
    while q:
        tmp = q
        q = []
        res = []
        for node in tmp:
            res.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        ans.append(res)
    return ans[::-1]


def solution_429(root: Optional[Node]) -> List[List[int]]:
    if not root:
        return []
    q = [root]
    ans = []
    while q:
        tmp = q
        q = []
        vals = []
        for node in tmp:
            vals.append(node.val)
            for child in node.children:
                q.append(child)
        ans.append(vals)
    return ans


def solution_105(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    f = {val: idx for idx, val in enumerate(inorder)}
    n = len(preorder)

    def helper(pre_left, pre_right, in_left, in_right):
        if pre_left > pre_right:
            return None
        p = TreeNode(preorder[pre_left])
        if pre_left == pre_right:
            return p
        idx = f[p.val]
        length = idx - in_left
        p.left = helper(pre_left + 1, pre_left + length, in_left, idx - 1)
        p.right = helper(pre_left + length + 1, pre_right, idx + 1, in_right)
        return p

    return helper(0, n - 1, 0, n - 1)


def solution_105_2(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    root = TreeNode(preorder[0])
    s = [root]
    index = 0
    cur = inorder[index]
    for val in preorder[1:]:
        node = s[-1]
        if node.val == cur:
            while s and s[-1].val == cur:
                index += 1
                cur = inorder[index]
                node = s.pop()
            node.right = TreeNode(val)
            s.append(node.right)
        else:
            node.left = TreeNode(val)
            s.append(node.left)
    return root


def solution_106(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    f = {x: i for i, x in enumerate(inorder)}
    n = len(postorder)

    def helper(i_left, i_right, p_left, p_right):
        if i_left > i_right:
            return None
        p = TreeNode(postorder[p_right])
        if i_left == i_right:
            return p
        idx = f[postorder[p_right]]
        length = idx - i_left
        p.left = helper(i_left, idx - 1, p_left, p_left + length - 1)
        p.right = helper(idx + 1, i_right, p_left + length, p_right - 1)
        return p

    return helper(0, n - 1, 0, n - 1)


def solution_106_2(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    root = TreeNode(postorder[-1])
    s = [root]
    idx = len(inorder) - 1
    cur = inorder[idx]
    for val in reversed(postorder[:-1:]):
        node = s[-1]
        if node.val == cur:
            while s and s[-1].val == cur:
                idx -= 1
                cur = inorder[idx]
                node = s.pop()
            node.left = TreeNode(val)
            s.append(node.left)
        else:
            node.right = TreeNode(val)
            s.append(node.right)
    return root


def solution_235(root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root or root == q or root == p:
        return root
    left = solution_235(root.left, p, q)
    right = solution_235(root.right, p, q)
    if left and right:
        return root
    if left:
        return left
    if right:
        return right
    return None


def solution_235_2(root: Optional[TreeNode], p: Optional[TreeNode], q: Optional[TreeNode]) -> Optional[TreeNode]:
    cur = root
    while cur:
        val = cur.val
        if val > p.val and val > q.val:
            cur = cur.left
            continue
        if val < p.val and val < q.val:
            cur = cur.right
            continue
        return cur
    return cur


def solution_257(root: Optional[TreeNode]) -> List[str]:
    paths = []

    def dfs(node, path):
        path += "->" + str(node.val)
        if not node.left and not node.right:
            if len(path) > 0:
                paths.append(path[2:])
            return
        if node.left:
            dfs(node.left, path)
        if node.right:
            dfs(node.right, path)

    dfs(root, "")
    return paths


def solution_258(num: int) -> int:
    while num >= 10:
        total = 0
        while num > 0:
            total += num % 10
            num = num // 10
        num = total
    return num


def solution_258_2(num: int) -> int:
    return (num - 1) % 9 + 1 if num else 0


def solution_263(n: int) -> bool:
    if n <= 0:
        return False
    # if n == 1:
    #     return True
    # while n != 1:
    #     if n % 5 == 0:
    #         n = n // 5
    #         continue
    #     if n % 3 == 0:
    #         n = n // 3
    #         continue
    #     if n % 2 == 0:
    #         n = n // 2
    #         continue
    #     return False
    # return True
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1


def solution_268(nums: List[int]) -> int:
    n = len(nums)
    return sum(range(n + 1)) - sum(nums)


def solution_15(nums: List[int]) -> List[List[int]]:
    f = {}
    ans = set()
    for num in nums:
        if num in f:
            f[num] += 1
        else:
            f[num] = 1
    for num1 in f:
        f[num1] -= 1
        for num2 in f:
            if f[num2] <= 0:
                continue
            f[num2] -= 1
            target = -num1 - num2
            if target in f and f[target] > 0:
                tmp = [target, num1, num2]
                tmp.sort()
                ans.add((tmp[0], tmp[1], tmp[2]))
            f[num2] += 1
        f[num1] += 1
    res = []
    for x in ans:
        res.append([x[0], x[1], x[2]])
    return res


def solution_15_2(nums: List[int]) -> List[List[int]]:
    n = len(nums)
    nums.sort()
    ans = []
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if sum(nums[i:i + 3]) > 0:  # 后面不可能为0
            break
        if nums[i] + nums[-2] + nums[-1] < 0:  # 这个位置不可能为0
            continue
        target = - nums[i]
        p, q = i + 1, n - 1
        while p < q:
            cur = nums[p] + nums[q]
            if cur < target:
                p += 1
            elif cur > target:
                q -= 1
            else:
                ans.append([nums[i], nums[p], nums[q]])
                # 可能还有答案
                p += 1
                while p < q and nums[p] == nums[p - 1]:
                    p += 1
                q -= 1
                while p < q and nums[q] == nums[q + 1]:
                    q -= 1
    return ans


def solution_167(numbers: List[int], target: int) -> List[int]:
    n = len(numbers)
    p, q = 0, n - 1
    while p < q:
        cur = numbers[p] + numbers[q]
        if cur < target:
            p += 1
        elif cur > target:
            q -= 1
        else:
            return [p + 1, q + 1]


def solution_18(nums: List[int], target: int) -> List[List[int]]:
    # no optimize
    nums.sort()
    n = len(nums)
    ans = []
    for i in range(n - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, n - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            p, q = j + 1, n - 1
            while p < q:
                cur = nums[p] + nums[q] + nums[i] + nums[j]
                if cur == target:
                    ans.append([nums[i], nums[j], nums[p], nums[q]])
                    p += 1
                    while p < q and nums[p] == nums[p - 1]:
                        p += 1
                    q -= 1
                    while p < q and nums[q] == nums[q + 1]:
                        q -= 1
                elif cur > target:
                    q -= 1
                    continue
                else:
                    p += 1
                    continue
    return ans


def solution_18_2(nums: List[int], target: int) -> List[List[int]]:
    # 剪枝
    nums.sort()
    n = len(nums)
    ans = []
    for i in range(n - 3):
        if sum(nums[i:i + 4]) > target:
            break
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        if nums[i] + nums[-3] + nums[-2] + nums[-1] < target:
            continue
        for j in range(i + 1, n - 2):
            if sum(nums[j:j + 3]) + nums[i] > target:
                break
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            if nums[i] + nums[j] + nums[-2] + nums[-1] < target:
                continue
            p, q = j + 1, n - 1
            while p < q:
                cur = nums[p] + nums[q] + nums[i] + nums[j]
                if cur == target:
                    ans.append([nums[i], nums[j], nums[p], nums[q]])
                    p += 1
                    while p < q and nums[p] == nums[p - 1]:
                        p += 1
                    q -= 1
                    while p < q and nums[q] == nums[q + 1]:
                        q -= 1
                elif cur > target:
                    q -= 1
                    continue
                else:
                    p += 1
                    continue
    return ans


def solution_278(n: int, isBadVersion: Callable[[int], bool]) -> int:
    left, right = -1, n + 1
    while left + 1 < right:
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid
    return right


def solution_283(nums: List[int]) -> None:
    n = len(nums)
    # while p_zero < n and nums[p_zero] != 0:
    #     p_zero += 1
    # p_non_zero = p_zero + 1
    # while p_non_zero < n and nums[p_non_zero] == 0:
    #     p_non_zero += 1
    # if p_non_zero > n:
    #     return
    p_zero, p_non_zero = 0, 0
    while p_zero < n and p_non_zero < n:
        while p_zero < n and nums[p_zero] != 0:
            p_zero += 1
        p_non_zero = max(p_non_zero, p_zero)
        while p_non_zero < n and nums[p_non_zero] == 0:
            p_non_zero += 1
        if p_non_zero >= n:
            break
        nums[p_non_zero], nums[p_zero] = 0, nums[p_non_zero]


def solution_303():
    # 前缀和
    # data_struct#NumArray
    ...


def solution_326(n: int) -> bool:
    if n < 1:
        return False
    while n != 1:
        if n % 3 == 0:
            n //= 3
        else:
            return False
    return True
    # return n > 0 && 1162261467 % n == 0;
    # 1162261467 = 3**19


def solution_11(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    area = 0
    while l < r:
        if (cur := height[l]) <= height[r]:
            area = max(area, (r - l) * (min(height[l], height[r])))
            l += 1
            while height[l] < cur and l < r:
                l += 1  # 面积不可能变大
        else:
            cur = height[r]
            area = max(area, (r - l) * (min(height[l], height[r])))
            r -= 1
            while height[r] < cur and l < r:
                r -= 1  # 面积不可能变大
    return area


def solution_42(height: List[int]) -> int:
    # 前后缀分解
    n = len(height)
    pre = [0] * n
    suf = [0] * n
    mp = 0
    ms = 0
    for i in range(n):
        mp = max(mp, height[i])
        pre[i] = mp
    for i in range(n - 1, -1, -1):
        ms = max(ms, height[i])
        suf[i] = ms
    ans = 0
    for i in range(n):
        ans += min(pre[i], suf[i]) - height[i]
    return ans


def solution_42_2(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    ml = height[l]
    mr = height[r]
    ans = 0
    while l < r:  # l=r必定是最高峰
        if ml < mr:
            ans += ml - height[l]  # ml一定大于等于height[l],因为ml先更新为最大值
            l += 1
            if l < r:
                ml = max(ml, height[l])
        else:
            ans += mr - height[r]
            r -= 1
            if l < r:
                mr = max(mr, height[r])
    return ans


def solution_42_3(height: List[int]) -> int:
    s = []
    ans = 0
    for i, h in enumerate(height):
        while s and h >= height[s[-1]]:  # 单调递减栈
            bottom_h = height[s.pop()]  # 栈顶为最小值
            if not s:
                break
            left = s[-1]  # 次小值
            dh = min(height[left], h) - bottom_h
            ans += dh * (i - left - 1)
        s.append(i)
    return ans


def solution_299(secret: str, guess: str) -> str:
    cnt_s = defaultdict(int)
    cnt_g = defaultdict(int)
    n = len(secret)
    for i in range(n):
        cnt_s[secret[i]] += 1
        cnt_g[guess[i]] += 1
    cow = 0
    for num in cnt_s:
        if num in cnt_g:
            cow += min(cnt_g[num], cnt_s[num])
    bull = 0
    for i in range(n):
        if secret[i] == guess[i]:
            bull += 1
    return str(bull) + 'A' + str(cow - bull) + 'B'


def solution_338(n: int) -> List[int]:
    # DP 最高有效位
    high_bit = 0
    i = 1
    ans = [0] * (n + 1)
    while i <= n:
        if i & (i - 1) == 0:
            high_bit = i
        ans[i] = ans[i - high_bit] + 1
        i += 1
    return ans


def solution_338_2(n: int) -> List[int]:
    # DP 最低有效位
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i >> 1] + (1 & i)
    return ans


def solution_342(n: int) -> bool:
    # 如果 n 是4的幂
    # 4的幂一定是2的幂
    # 区分
    # 2的幂却不是4的幂 => 4^x * 2 模3余2
    # 4^x => (3+1)^x 与 1^x => 1 对3同余
    if n <= 0:
        return False
    if (n & (n - 1) == 0) and n % 3 == 1:
        return True
    else:
        return False


def solution_342_2(n: int) -> bool:
    mask = 0xaaaaaaaa  # 1010 1010 1010 1010 1010 1010 1010 1010
    # mask = 0x2aaaaaaa # 0010 1010 1010 1010 1010 1010 1010 1010 忽略负数
    return n > 0 and (n & (n - 1)) == 0 and n & mask == 0
    # n正 n是2的幂 n只有偶数位为1


def solution_344(s: List[str]) -> None:
    n = len(s)
    p, q = 0, n - 1
    while p < q:
        s[p], s[q] = s[q], s[p]
        p += 1
        q -= 1
    return


def solution_345(s: str) -> str:
    ans = list(s)
    n = len(ans)
    a = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', }
    p, q = 0, n - 1
    while p < q:
        while p < n and ans[p] not in a:
            p += 1
        while q > -1 and ans[q] not in a:
            q -= 1
        if p < q:
            ans[p], ans[q] = ans[q], ans[p]
            p += 1
            q -= 1
    return ''.join(ans)
