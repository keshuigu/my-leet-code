import heapq
from bisect import bisect_left

from .data_struct import *
from .method import *


def solution_2235(num1: int, num2: int) -> int:
    return num1 + num2


def solution_2487(head: Optional[ListNode]) -> Optional[ListNode]:
    # 双向链表
    if head is None or head.next is None:
        return head
    d_head = DListNode(head)
    p = head
    q = d_head
    while p.next is not None:
        q.next = DListNode(p.next)
        q.next.prev = q
        q = q.next
        p = p.next
    head = q.val
    q = q.prev
    while q is not None:
        if q.val.val >= head.val:
            temp = head
            head = q.val
            head.next = temp
        q = q.prev
    return head


def solution_2487_2(head: Optional[ListNode]) -> Optional[ListNode]:
    # 反转链表
    def reverse(head_re: ListNode) -> ListNode:
        dummy = ListNode(head_re.val)
        while head_re is not None:
            p_re = head_re
            head_re = head_re.next
            p_re.next = dummy.next
            dummy.next = p_re
        return dummy.next

    head = reverse(head)
    # 找左侧有更大数值的节点删除
    p = head
    while p is not None and p.next is not None:
        if p.next.val < p.val:
            p.next = p.next.next
        else:
            p = p.next
    return reverse(head)


def solution_2487_3(head: Optional[ListNode]) -> Optional[ListNode]:
    # 单调栈
    stack = []
    p = head
    rest = ListNode(-1)
    while p is not None:
        while not len(stack) == 0:
            if p.val > stack[len(stack) - 1].val:
                stack.pop()
            else:
                stack[len(stack) - 1].next = p
                stack.append(p)
                break
        if len(stack) == 0:
            stack.append(p)
            rest.next = p
        p = p.next
    return rest.next


def solution_2397(matrix: List[List[int]], numSelect: int) -> int:
    com_arrays = com_iteration(len(matrix[0]), numSelect)
    count_arrays = []
    for com_item in com_arrays:
        matrix_copy = []
        for i in range(len(matrix)):
            matrix_copy.append(list(matrix[i]))
        for i in range(len(matrix_copy)):
            for j in range(len(matrix_copy[0])):
                if j + 1 in com_item:
                    matrix_copy[i][j] = 0
        count = 0
        for row in matrix_copy:
            if sum(row) == 0:
                count += 1
        count_arrays.append(count)
    return max(count_arrays)


def solution_2397_2(matrix: List[List[int]], numSelect: int) -> int:
    # 预处理矩阵,压缩成m*1的数组,用二进制数表示
    m = len(matrix)
    n = len(matrix[0])
    mask = [0] * m
    for i in range(m):
        for j in range(n):
            mask[i] += matrix[i][j] << (n - j - 1)
    # 用一个整数来表述所有组合,不满足的跳过即可
    limit = 1 << n
    res = 0
    for i in range(limit):
        if pop_count(i) != numSelect:
            continue
        count = 0
        for j in range(m):
            if mask[j] & i == mask[j]:
                count += 1
        res = max(res, count)
    return res


def solution_2397_3(matrix: List[List[int]], numSelect: int) -> int:
    # 预处理矩阵,压缩成m*1的数组,用二进制数表示
    m = len(matrix)
    n = len(matrix[0])
    mask = [0] * m
    for i in range(m):
        for j in range(n):
            mask[i] += matrix[i][j] << (n - j - 1)
    # Gosper's Hack
    res = 0
    limit = 1 << n
    cur = (1 << numSelect) - 1
    while cur < limit:
        count = 0
        for j in range(m):
            if mask[j] & i == mask[j]:
                count += 1
        res = max(res, count)
        lb = cur & -cur  # 取最低位的1
        r = cur + lb  # 在cur最低位的1上加1
        cur = ((r ^ cur) >> count_trailing_zeros(lb) + 2) | r
    return res


def solution_2085(words1: List[str], words2: List[str]) -> int:
    f = {}
    for word in words1:
        if word not in f:
            f[word] = 1
        else:
            f[word] += 1
    for word in f:
        if f[word] > 1:
            f[word] = 3
    for word in words2:
        if word in f:
            f[word] += 1
    count = 0
    for word in f:
        if f[word] == 2:
            count += 1
    return count


def solution_2182(s: str, repeatLimit: int) -> str:
    nums_ch = [0] * 26
    ord_a = ord('a')
    for ch in s:
        nums_ch[ord(ch) - ord_a] += 1
    i = 25
    res = ''
    count = 0
    while i >= 0:
        if nums_ch[i] == 0:
            i -= 1
            continue
        else:
            res += chr(ord_a + i)
            nums_ch[i] -= 1
            count += 1
            if nums_ch[i] == 0:
                i -= 1
                count = 0
                continue
            if count == repeatLimit:
                j = i - 1
                while nums_ch[j] == 0:
                    j -= 1
                if j < 0:
                    return res
                res += chr(ord_a + j)
                nums_ch[j] -= 1
                count = 0
    return res


def solution_2182_2(s: str, repeatLimit: int) -> str:
    # 优先队列
    # 先存一下每个字符的次数
    pq = PriorityQueue()
    nums_ch = [0] * 26
    ord_a = ord('a')
    for ch in s:
        nums_ch[ord(ch) - ord_a] += 1
        if nums_ch[ord(ch) - ord_a] == 1:
            pq.put(ord(ch) - ord_a)  # 保证没有重复
    res = ''
    count = 0
    while not pq.empty():
        tmp = pq.delete()
        if tmp is None:
            return res
        res += chr(ord_a + tmp)
        count += 1
        nums_ch[tmp] -= 1
        if nums_ch[tmp] == 0:
            count = 0
            continue
        if count < repeatLimit:
            pq.put(tmp)
        else:
            sub_tmp = pq.delete()
            if sub_tmp is None:
                return res
            res += chr(ord_a + sub_tmp)
            count = 0
            nums_ch[sub_tmp] -= 1
            if nums_ch[sub_tmp] != 0:
                pq.put(sub_tmp)
            pq.put(tmp)
    return res


def solution_2376(n: int) -> int:
    s = str(n)
    # dp 需要的空间
    # M = s.length
    # n最大是个10位数
    # 二进制从低到高第 i 位为 1 表示 i 在集合中，为 0 表示 i 不在集合中。例如集合 {0,2,3} 对应的二进制数为 1101
    # 要求n各位互不相同，因此只需要9位就能表示所有的情况
    # 987654321 => 111111111 => (1<<10 -1)
    # dp = [[-1] * (1 << 10)] * len(s)# 重复指针，会出错
    dp = [[-1] * (1 << 10) for _ in range(len(s))]

    def dfs(i: int, mask: int, limit: bool, num: bool) -> int:
        # 末尾判断
        if i == len(s):
            return 1 if num else 0
        # 只有不受限制并且dp已经计算过的情况下，才能复用
        if not limit and num and dp[i][mask] != -1:
            return dp[i][mask]
        res = 0
        # 前导0，如果还没开始，比如n=999，现在检查9，那么就是009
        if not num:
            # 先考虑继续为0的情况
            # 如果高位取了0，一定比n小，limit必定False
            res += dfs(i + 1, mask, False, False)
        # 再考虑不是0的情况
        low = 0 if num else 1
        high = int(s[i]) if limit else 9  # 如果顶到上限了，那么这里检查的不能超过ｎ
        for x in range(low, high + 1):
            if ((mask >> x) & 1) == 0:
                res += dfs(i + 1, mask | (1 << x), limit and x == high, True)
        dp[i][mask] = res
        return res

    count = dfs(0, 0, True, False)
    return count


def solution_2376_2(n: int) -> int:
    return countSpecialNumbers(n)


def solution_2171(beans: List[int]) -> int:
    beans.sort()
    s = [beans[0]]
    n = len(beans)
    for i in range(1, n):
        s.append(beans[i] + s[i - 1])
    total = s[n - 1]
    min_beans = 10 ** 10
    for i in range(0, n):
        tmp = total - (n - i) * beans[i]
        if min_beans > tmp:
            min_beans = tmp
    return min_beans


def solution_2476(root: Optional[TreeNode], queries: List[int]) -> List[List[int]]:
    # TL
    ans = []
    for q in queries:
        p = root
        left = -1
        right = -1
        while p is not None:
            if q < p.val:
                right = p.val
                p = p.left
            elif q > p.val:
                left = p.val
                p = p.right
            else:
                left = right = q
                break
        ans.append([left, right])
    return ans


def solution_2476_2(root: Optional[TreeNode], queries: List[int]) -> List[List[int]]:
    a = []

    def dfs(node: Optional[TreeNode]) -> None:
        if node is None:
            return
        dfs(node.left)
        a.append(node.val)
        dfs(node.right)

    dfs(root)
    n = len(a)
    ans = []
    for q in queries:
        j = bisect_left(a, q)
        mx = a[j] if j < n else -1
        if j == n or a[j] != q:
            j -= 1
        mn = a[j] if j >= 0 else -1
        ans.append([mn, mx])
    return ans


def solution_2368(n: int, edges: List[List[int]], restricted: List[int]) -> int:
    g = [[] * n for _ in range(n)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)

    ans = 0
    s_r = set(restricted)

    def dfs(node, p):
        nonlocal ans
        ans += 1
        for x in g[node]:
            if x not in s_r and x != p:
                dfs(x, node)

    dfs(0, -1)
    return ans


def solution_2369(nums: List[int]) -> bool:
    n = len(nums)

    @cache
    def dfs(left):
        if left >= n - 1:
            return False
        if left + 2 == n:
            if nums[left] == nums[left + 1]:
                return True
            else:
                return False
        if left + 3 == n:
            if nums[left] == nums[left + 1] == nums[-1]:
                return True
            elif nums[left] + 1 == nums[left + 1] and nums[left + 1] + 1 == nums[left + 2]:
                return True
            else:
                return False
        res = False
        if nums[left] == nums[left + 1]:
            res = res or dfs(left + 2)
        if nums[left] == nums[left + 1] == nums[left + 2]:
            res = res or dfs(left + 3)
        if nums[left] + 1 == nums[left + 1] and nums[left + 1] + 1 == nums[left + 2]:
            res = res or dfs(left + 3)
        return res

    return dfs(0)


def solution_2369_2(nums: List[int]) -> bool:
    n = len(nums)
    f = [True] + [False] * n
    # nums 的前 i 个数能否有效划分
    for i, x in enumerate(nums):
        if (i > 0 and f[i - 1] and x == nums[i - 1]) or (
                i > 1 and f[i - 2] and (
                (x == nums[i - 1] == nums[i - 2]) or (x == nums[i - 1] + 1 == nums[i - 2] + 2))):
            f[i + 1] = True
    return f[n]


def solution_2386(nums: List[int], k: int) -> int:
    ms = 0
    for i, x in enumerate(nums):
        if x >= 0:
            ms += x
        else:
            nums[i] = -x
    nums.sort()
    h = [(0, 0)]  # 空子序列
    # Dijkstra
    for _ in range(k - 1):
        s, i = heapq.heappop(h)
        if i < len(nums):
            # 在子序列的末尾添加nums[i]
            heapq.heappush(h, (s + nums[i], i + 1))  # 下一个添加/替换的元素下标为 i+1
            if i:  # 替换子序列的末尾元素为 nums[i]
                heapq.heappush(h, (s + nums[i] - nums[i - 1], i + 1))
    return ms - h[0][0]


def solution_2386_2(nums: List[int], k: int) -> int:
    """
    二分答案 + dp
    # 判断是否有至少k个子序列，其元素和s不超过sum_limit
    """
    s = 0
    for i, x in enumerate(nums):
        if x >= 0:
            s += x
        else:
            nums[i] = -x
    nums.sort()

    def check(sum_limit: int) -> bool:
        cnt = 1  # 空序列

        def dfs(i: int, s: int) -> None:
            nonlocal cnt
            if cnt == k or i == len(nums) or s + nums[i] > sum_limit:
                return
            cnt += 1  # 选了i 子序列+1
            dfs(i + 1, s + nums[i])
            dfs(i + 1, s)

        dfs(0, 0)
        return cnt == k

    # 0到sum(nums)-1
    # 二分
    return s - bisect_left(range(sum(nums)), True, key=check)
