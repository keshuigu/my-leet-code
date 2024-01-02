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
