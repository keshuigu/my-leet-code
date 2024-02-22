from collections import defaultdict
from functools import cache

from .data_struct import *


def solution_876(head: Optional[ListNode]) -> Optional[ListNode]:
    # 快慢指针
    p = q = head
    while p is not None and p.next is not None:
        p = p.next.next
        q = q.next
    return q


def solution_600(n: int) -> int:
    s = bin(n)[2:]
    length = len(s)
    dp = [[-1] * 2 for _ in range(length)]

    def f(i, j, limit):
        if i == length:
            return 1
        if not limit and dp[i][j] != -1:
            return dp[i][j]
        res = 0
        # 复杂
        if j == 1:
            res += f(i + 1, 0, limit and s[i] == '0')
        else:
            res += f(i + 1, 0, limit and s[i] == '0')
            if s[i] == '1' or not limit:
                res += f(i + 1, 1, limit and s[i] == '1')
        # up = int(s[i]) if limit else 1
        # res += f(i + 1, 0, limit and up == 0)
        # if j == 0 and up == 1:
        #     res += f(i + 1, 1, limit)
        if not limit:
            dp[i][j] = res
        return res

    return f(0, 0, True)


def solution_902(digits: List[str], n: int) -> int:
    s = str(n)
    ld = len(digits)
    ls = len(s)
    dp = [-1] * ls
    my_set = set(digits)
    min_num = int(digits[0])
    max_num = int(digits[ld - 1])

    def dfs(i, limit, num):
        if i == ls:
            return 1 if num else 0
        if not limit and dp[i] != -1:
            return dp[i]
        res = 0
        if not num:
            res += dfs(i + 1, False, False)
        low = min_num
        up = int(s[i]) if limit else max_num
        for d in range(low, up + 1):
            if str(d) in my_set:
                res += dfs(i + 1, limit and d == up, True)
        if not limit and num:
            dp[i] = res
        return res

    return dfs(0, True, False)


def solution_670(num: int) -> int:
    s = str(num)
    chs = list(enumerate(s))
    chs.sort(reverse=True, key=lambda x: x[1])
    for i in range(len(chs)):
        index, ch = chs[i]
        if ch != s[i]:
            for j in range(i + 1, len(chs)):
                if chs[j][1] == ch:
                    index = chs[j][0]
                else:
                    break
            res = s[:i] + ch + s[i + 1:index] + s[i] + s[index + 1:]
            return int(res)
    return num


def solution_670_2(num: int) -> int:
    s = list(str(num))
    max_idx = len(s) - 1
    p = q = -1
    for i in range(len(s) - 2, -1, -1):
        if s[i] > s[max_idx]:
            max_idx = i
        elif s[i] < s[max_idx]:
            p, q = i, max_idx
    if p == -1:
        return num
    s[p], s[q] = s[q], s[p]
    return int(''.join(s))


def solution_514(ring: str, key: str) -> int:
    s = [ord(c) - ord('a') for c in ring]
    t = [ord(c) - ord('a') for c in key]
    n = len(s)
    # 先算出每个字母的最后一次出现的下标
    pos = [0] * 26
    for i, c in enumerate(s):
        pos[c] = i
    # 计算每个s[i]左边a-z的最近下标
    # pos保存的是从右边数的最后一个下标，因此也就是从左边数的第一个下标
    # 上面的循环结束刚好是s[0]对应的所有字母从左数的第一个下标
    # 更新pos[c]相当于更新这个环的起始位置，更新对于s[i+1]来言，最近的s[i]所在的位置
    # 其他字母的位置不会变化
    left = [None] * n
    for i, c in enumerate(s):
        left[i] = pos[:]
        pos[c] = i

    # 对右边的计算类似,需要倒序来计算最早出现的下标
    for i in range(n - 1, -1, -1):
        pos[s[i]] = i
    right = [None] * n
    for i in range(n - 1, -1, -1):
        right[i] = pos[:]
        pos[s[i]] = i

    # 对于当前s[i],旋转到t[j]所需要的最小开销
    @cache
    def dfs(i: int, j: int) -> int:
        if j == len(t):
            return 0
        c = t[j]
        if s[i] == c:
            return dfs(i, j + 1)
        # 左侧还是右侧最小值
        l, r = left[i][c], right[i][c]
        return min(dfs(l, j + 1) + ((n - l + i) if l > i else i - l),
                   dfs(r, j + 1) + ((n - i + r) if r < i else r - i))

    return dfs(0, 0) + len(t)


def solution_514_2(ring: str, key: str) -> int:
    s = [ord(c) - ord('a') for c in ring]
    t = [ord(c) - ord('a') for c in key]
    n = len(s)
    # 先算出每个字母的最后一次出现的下标
    pos = [0] * 26
    for i, c in enumerate(s):
        pos[c] = i
    # 计算每个s[i]左边a-z的最近下标
    # pos保存的是从右边数的最后一个下标，因此也就是从左边数的第一个下标
    # 上面的循环结束刚好是s[0]对应的所有字母从左数的第一个下标
    # 更新pos[c]相当于更新这个环的起始位置，更新对于s[i+1]来言，最近的s[i]所在的位置
    # 其他字母的位置不会变化
    left = [None] * n
    for i, c in enumerate(s):
        left[i] = pos[:]
        pos[c] = i

    # 对右边的计算类似,需要倒序来计算最早出现的下标
    for i in range(n - 1, -1, -1):
        pos[s[i]] = i
    right = [None] * n
    for i in range(n - 1, -1, -1):
        right[i] = pos[:]
        pos[s[i]] = i

    # 对于当前s[i],旋转到t[j]所需要的最小开销
    memo = [[-1] * n for _ in range(len(t))]

    def dfs(i: int, j: int) -> int:
        if j == len(t):
            return 0
        if memo[j][i] != -1:
            return memo[j][i]
        c = t[j]
        if s[i] == c:
            memo[j][i] = dfs(i, j + 1)
            return memo[j][i]
        # 左侧还是右侧最小值
        l, r = left[i][c], right[i][c]
        memo[j][i] = min(dfs(l, j + 1) + ((n - l + i) if l > i else i - l),
                         dfs(r, j + 1) + ((n - i + r) if r < i else r - i))
        return memo[j][i]

    return dfs(0, 0) + len(t)


def solution_993(root: Optional[TreeNode], x: int, y: int) -> bool:
    q = [root]
    while q:
        tmp = q
        q = []
        index = -1
        x_index = -1
        y_index = -1
        for node in tmp:
            index += 1
            if node.left:
                q.append(node.left)
                if node.left.val == x:
                    x_index = index
                if node.left.val == y:
                    y_index = index
            index += 1
            if node.right:
                q.append(node.right)
                if node.right.val == x:
                    x_index = index
                if node.right.val == y:
                    y_index = index
        if x_index == -1 and y_index == -1:
            continue
        if x_index == -1 or y_index == -1:
            return False
        elif abs(x_index - y_index) == 1 and ((x_index + y_index) // 2) % 2 == 0:
            return False
        else:
            return True


def solution_987(root: Optional[TreeNode]) -> List[List[int]]:
    f = defaultdict(list)

    def helper(p: TreeNode, row, col):
        if not p:
            return
        f[(row, col)].append(p.val)
        helper(p.left, row + 1, col - 1)
        helper(p.right, row + 1, col + 1)

    helper(root, 0, 0)
    keys = []
    for key in f:
        keys.append(key)
        f[key] = sorted(f[key])
    keys.sort(key=lambda p: (p[1], p[0]))
    cur_col = keys[0][1]
    index = 0
    ans = [[]]
    for key in keys:
        if key[1] == cur_col:
            ans[index].extend(f[key])
        else:
            cur_col = key[1]
            index += 1
            ans.append(f[key])
    return ans


def solution_589(root: Optional[Node]) -> List[int]:
    if not root:
        return []
    res = []

    def helper(p):
        res.append(p.val)
        for child in p.children:
            helper(child)

    helper(root)
    return res


def solution_589_2(root: Optional[Node]) -> List[int]:
    if not root:
        return []
    ans = []
    s = [root]
    while s:
        cur = s.pop()
        ans.append(cur.val)
        for child in cur.children[::-1]:
            s.append(child)
    return ans


def solution_590(root: Optional[Node]) -> List[int]:
    if not root:
        return []
    ans = []

    def helper(node):
        for child in node.children:
            helper(child)
        ans.append(node.val)

    helper(root)
    return ans


def solution_590_2(root: Optional[Node]) -> List[int]:
    if not root:
        return []
    ans = []
    s = [root]
    prev = None
    while s:
        cur = s.pop()
        if not cur.children or prev == cur.children[-1]:
            ans.append(cur.val)
            prev = cur
            continue
        s.append(cur)
        for child in cur.children[::-1]:
            s.append(child)
    return ans


def solution_889(preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    f = {x: i for i, x in enumerate(postorder)}

    def helper(pre_l, pre_r, post_l, post_r):
        if pre_l > pre_r or post_l > post_r:
            return None
        node = TreeNode(preorder[pre_l])
        if pre_r == pre_l:
            return node
        left_val = preorder[pre_l + 1]
        idx = f[left_val]
        length = idx - post_l + 1
        node.left = helper(pre_l + 1, pre_l + length, post_l, post_l + length - 1)
        node.right = helper(pre_l + length + 1, pre_r, post_l + length, post_r - 1)
        return node

    n = len(postorder)
    return helper(0, n - 1, 0, n - 1)
