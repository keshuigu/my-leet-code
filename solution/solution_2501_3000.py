import itertools
from collections import defaultdict
from math import isqrt

from .data_struct import *
from .method import *


def solution_2706(prices: List[int], money: int) -> int:
    min_1 = prices[0] if prices[0] <= prices[1] else prices[1]
    min_2 = prices[1] if prices[0] <= prices[1] else prices[0]
    for price in prices[2:]:
        if price < min_1:
            min_2 = min_1
            min_1 = price
        elif min_1 <= price < min_2:
            min_2 = price
    res = money - min_1 - min_2
    return money if res < 0 else res


def solution_2807(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head
    p, q = head, head.next
    while p and q:
        tmp = gcd_euclid(p.val, q.val)
        p.next = ListNode(tmp)
        p.next.next = q
        p = q
        q = q.next
    return head


def solution_2707(s: str, dictionary: List[str]) -> int:
    """
    设 n 是 s 的长度，现在有两种基本的分割方案：

    把 s 的最后一个字符 s[n−1] 当做是额外字符，那么问题转为长度为 n−1 的子问题。
    找到一个 j 使得 s 的后缀 s[j...n−1] 构成的子串在 dictionary，那么问题转为长度为 j 的子问题。
    因此，定义 d[i] 为 sss 前缀 s[0...i−1] 的子问题，那么 d[i] 取下面两种情况的最小值：
    1. 把 s[i−1]当做是额外字符，d[i]=d[i−1]+1
    2. 遍历所有的 j(j∈[0,i−1])，如果子字符串 s[j...i−1]存在于 dictionary 中，那么 d[i]=mind[j]

    初始状态 d[0]=0d[0] = 0d[0]=0，最终答案为 d[n]d[n]d[n]。
    查找子串 s[j...i−1]s[j...i-1]s[j...i−1] 是否存在于 dictionary 可以使用哈希表。
    另外在实现动态规划时，可以使用记忆化搜索，也可以使用递推，这两种方式在时空复杂度方面并没有明显差异。
    """
    n = len(s)
    dp = [0] * (n + 1)  # dp[i]代表s[0:i]
    trie = Trie()
    for tmp in dictionary:
        trie.insert(tmp[::-1])
    for i in range(1, n + 1):
        dp[i] = dp[i - 1] + 1
        node = trie
        for j in range(i - 1, -1, -1):  # 逆序遍历i-1到0
            node, ok = track(node, s[j])
            if ok:
                dp[i] = min(dp[i], dp[j])
    return dp[n]


def solution_2696(s: str) -> int:
    stack = []
    for ch in s:
        if not stack:
            stack.append(ch)
        elif (ch == 'B' and stack[-1] == 'A') or (ch == 'D' and stack[-1] == 'C'):
            stack.pop()
        else:
            stack.append(ch)
    return len(stack)


def solution_2645(word: str) -> int:
    p, q, count = 0, 0, 0
    pattern = 'abc'
    while p < len(word):
        if word[p] != pattern[q]:
            count += 1
        else:
            p += 1
        q = (q + 1) % 3
    count += (3 - q) % 3
    return count


def solution_2645_2(word: str) -> int:
    # dp
    # d[i] = min(d[i]+2, d[i-1] -1)
    # 第二种情况需要word[i-1] > word[i-2],也就是word[i]是排在word[i-1]后面的字母,从而构成abc串
    n = len(word)
    d = [0] * (n + 1)
    d[1] = d[0] + 2
    for i in range(2, n + 1):
        d[i] = d[i - 1] + 2
        if word[i - 1] > word[i - 2]:
            d[i] = d[i - 1] - 1
    return d[n]


def solution_2645_3(word: str) -> int:
    # 直接拼接
    # 两个相邻位置之间插入字符数量
    # (word[i] - word[i-1] -1 + 3) mod 3
    # 头尾额外处理
    n = len(word)
    # word[0] 前 word[0]- 'a'
    # word[n-1]后 'c' - word[n-1]
    count = ord(word[0]) - ord(word[n - 1]) + 2
    for i in range(1, n):
        count += (ord(word[i]) - ord(word[i - 1]) + 2) % 3
    return count


def solution_2645_4(word: str) -> int:
    # 直接计算
    # 最终组数等于所有满足后者字符小于等于前者字符的情况+1
    n = len(word)
    count = 1
    for i in range(1, n):
        if word[i] <= word[i - 1]:
            count += 1
    return count * 3 - n


def solution_2719(num1: str, num2: str, min_sum: int, max_sum: int) -> int:
    # 超时
    # def get_sum(tmp: int) -> int:
    #     count = 0
    #     while tmp > 0:
    #         count += tmp % 10
    #         tmp = tmp // 10
    #     return count
    #
    # # f(num2, min_sum, max_sum) - f(num-1, min_sum, max_sum)
    # n1 = int(num1)
    # n2 = int(num2)
    # count_dp = 0
    # for i in range(0, n1):
    #     cur = get_sum(i)
    #     if min_sum <= cur <= max_sum:
    #         count_dp += 1
    # count_n1 = count_dp
    # for i in range(n1, n2 + 1):
    #     cur = get_sum(i)
    #     if min_sum <= cur <= max_sum:
    #         count_dp += 1
    # return (count_dp-count_n1) % 1000000007
    # 数位DP
    # d[][]表示还剩第i位到第0位的数字未填，而已填的数字位数之和为j时，符合条件的数字有多少个
    # d[0][j]表示还剩第0位的数字未填，而已填的数字位数之和为j时，符合条件的数字有多少个
    # 显然d[0][j]取决于第0位加上j的值是否满足条件，那么取值范围是0-10
    # d的记忆化过程是从i=0开始，i变大，j变小，但是每个状态只会计算一次
    N, M = 23, 401  # 限定d的空间大小
    MOD = 10 ** 9 + 7
    d = [[-1] * M for _ in range(N)]  # 预定义默认值

    def dfs(num, i, j, limit) -> int:
        if j > max_sum:  # 剪枝，如果j已经比max_sum大了，可以排除
            return 0
        if i == -1:  # 如果i遍历到-1，说明数字已经检查完了，j如果大于min_sum就可以记为1个有效的数字
            return j >= min_sum
        if not limit and d[i][j] != -1:  # 如果不是limit数，且已经存储了结果，那么返回对应的数即可
            return d[i][j]
        res = 0
        up = ord(num[i]) - ord('0') if limit > 0 else 9
        for x in range(up + 1):
            res = (res + dfs(num, i - 1, j + x, limit and x == up)) % MOD
        if not limit:
            d[i][j] = res
        return res

    def get(num):
        num = num[::-1]  # 题解判断方案是高位开始到低位，因此反转数字，使得n-1位对应d[n-1]
        return dfs(num, len(num) - 1, 0, True)

    # 简单得到num1-1
    def sub(num):
        i = len(num) - 1
        arr = list(num)
        while arr[i] == '0':
            i -= 1
        arr[i] = chr(ord(arr[i]) - 1)
        i += 1
        while i < len(num):
            arr[i] = '9'
            i += 1
        return ''.join(arr)

    return (get(num2) - get(sub(num1)) + MOD) % MOD


def solution_2744(words: List[str]) -> int:
    my_set = set(words)
    count = 0
    for word in words:
        if word[::-1] == word:
            continue
        if word[::-1] in my_set:
            count += 1
    return count // 2


def solution_2744_2(words: List[str]) -> int:
    # 只遍历一次
    my_set = set()
    count = 0
    for word in words:
        if word in my_set:
            count += 1
        my_set.add(word[::-1])
    return count


def solution_2809(nums1: List[int], nums2: List[int], x: int) -> int:
    def my_sort(left: int, right: int):
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
        my_sort(left, i - 1)
        my_sort(i + 1, right)

    n = len(nums1)
    my_sort(0, n - 1)
    # dp[i][j]:对前i个元素进行j次操作,所能减少的total的值
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    # sum = sum(nums1)+sum(nums2)*t-dp[n][t]
    # dp[i][j] = max(dp[i-1][j] , dp[i-1][j-1]+nums1[i-1]+nums2[i-1]*j)
    # 做操作肯定比不做要小
    sum1 = sum(nums1)
    sum2 = sum(nums2)
    for i in range(1, n + 1):
        for j in range(1, i + 1):
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - 1] + nums1[i - 1] + nums2[i - 1] * j)
        # 不能在循环里判断
        # dp[i][i]只是移除了所有前i个元素,并不是最小的值
        # 最小值可能存在于移除其他元素,但只用了i次
        # 因此要在遍历结束后再找答案
    for i in range(0, n + 1):
        if sum2 * i + sum1 - dp[n][i] <= x:
            return i
    return -1


def solution_2788(words: List[str], separator: str) -> List[str]:
    res = []
    for word in words:
        wl = word.split(separator)
        for w in wl:
            if len(w) > 0:
                res.append(w)
    return res


def solution_2788_2(words: List[str], separator: str) -> List[str]:
    res = []
    for word in words:
        last = 0
        for i in range(len(word)):
            if word[i] == separator:
                if i != last:
                    res.append(word[last:i])
                last = i + 1
        if last < len(word):
            res.append(word[last:])
    return res


def solution_2765(nums: List[int]) -> int:
    p, q = 0, 1
    cnt = 0
    max_cnt = 0
    while p < len(nums) and q < len(nums):
        if nums[q] - nums[p] == 1:
            cnt += 1
        else:
            max_cnt = max(max_cnt, cnt)
            if p > q:
                p, q = q, p
            if nums[q] - nums[p] == 1:
                cnt = 1
            else:
                cnt = 0
                p += 1
                q += 1
                continue
        if q < p:
            q += 2
        else:
            p += 2
    max_cnt = max(max_cnt, cnt)
    if max_cnt < 1:
        return -1
    else:
        return max_cnt + 1


def solution_2765_2(nums: List[int]) -> int:
    ans = -1
    i, n = 0, len(nums)
    while i < n - 1:
        if nums[i + 1] - nums[i] != 1:
            i += 1
            continue
        i0 = i
        i += 2
        while i < n and nums[i] == nums[i0] + (i - i0) % 2:
            i += 1
        ans = max(ans, i - i0)
        i -= 1  # 末尾有数重叠 3434 4545 4重叠
    return ans


def solution_2865(maxHeights: List[int]) -> int:
    n = len(maxHeights)
    heights = [0] * n
    sum_height = 0
    for i in range(n):
        cur_height = maxHeights[i]
        heights[i] = cur_height
        for j in range(i - 1, -1, -1):
            cur_height = min(maxHeights[j], cur_height)
            heights[j] = cur_height
        cur_height = heights[i]
        for k in range(i + 1, n):
            cur_height = min(maxHeights[k], cur_height)
            heights[k] = cur_height
        sum_height = max(sum_height, sum(heights))
    return sum_height


def solution_2865_2(maxHeights: List[int]) -> int:
    """
    单调栈的思路:
    先不考虑i在什么位置,用单调栈的思想遍历maxHeights,求得suf或者pre
    再从另一个方向走一遍,这一遍记录pre+suf,并逐渐更新答案
    """
    n = len(maxHeights)
    suf = [0] * (n + 1)  # 注意长度,额外的0表示最右边的塔最高
    stack = [n]  # 放一个哨兵节点,减少边界判断
    sum = 0  # 维护一个sum
    for i in range(n - 1, -1, -1):
        while len(stack) > 1 and maxHeights[i] <= maxHeights[stack[-1]]:  # 峰顶右侧离峰顶近的高度比远的小,那么弹出
            j = stack.pop()
            sum -= maxHeights[j] * (stack[-1] - j)  # 这一部分原本都是maxHeights[j],现在要更新成更小的值了
        sum += maxHeights[i] * (stack[-1] - i)  # 哨兵保证这里的stack[-1] - i是有意义的
        suf[i] = sum
        stack.append(i)  # 入栈
    ans = 0
    stack = [-1]
    sum = 0
    for i in range(n):
        while len(stack) > 1 and maxHeights[i] <= maxHeights[stack[-1]]:
            j = stack.pop()
            sum -= maxHeights[j] * (j - stack[-1])
        sum += maxHeights[i] * (i - stack[-1])
        ans = max(ans, sum + suf[i + 1])
        stack.append(i)
    return ans


def solution_2859(nums: List[int], k: int) -> int:
    n = len(nums)
    indexes = []
    # min_k = (1 << k) - 1
    tmp, cnt = n, 0
    while tmp > 0:
        cnt += 1
        tmp >>= 1
    comb = itertools.combinations(range(cnt + 1), k)
    for c in comb:
        index = 0
        for i in c:
            index += 1 << i
        if index < n:
            indexes.append(index)
    return sum([nums[x] for x in indexes])


def solution_2859_2(nums: List[int], k: int) -> int:
    # gospers_hack
    if k == 0:
        return nums[0]
    n = len(nums)
    tmp, cnt = n, 0
    while tmp > 0:
        cnt += 1
        tmp >>= 1
    # comb = itertools.combinations(range(cnt + 1), k)
    comb = []
    cur = (1 << k) - 1
    limit = 1 << cnt
    while cur < limit:
        comb.append(cur)
        lb = cur & -cur
        r = cur + lb
        cur = ((r ^ cur) >> ((lb & -lb).bit_length() - 1) + 2) | r
    return sum([nums[x] for x in comb if x < n])


def solution_2859_3(nums: List[int], k: int) -> int:
    # pop_count
    ans = 0
    for i in range(len(nums)):
        if pop_count(i) == k:
            ans += nums[i]
    return ans


def solution_2846(n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
    m = n.bit_length()
    g = [[] for _ in range(n)]
    for x, y, w in edges:
        g[x].append([y, w])
        g[y].append([x, w])
    freq = [[0] * 27 for _ in range(n)]
    depth = [0] * n
    pa = [[-1] * m for _ in range(n)]

    def dfs(n, p):
        pa[n][0] = p
        depth[n] = depth[p] + 1
        for y, w in g[n]:
            if y != p:
                # freq[y][w] += freq[n][w] + 1
                for k in range(27):
                    if k == w:
                        freq[y][w] = freq[n][w] + 1
                    else:
                        freq[y][k] = freq[n][k]
                dfs(y, n)

    dfs(0, -1)
    for i in range(m - 1):
        for x in range(n):
            if (p := pa[x][i]) != -1:
                pa[x][i + 1] = pa[p][i]

    def get_kth(x, k):
        for i in range(k.bit_length()):
            if (k >> i) & 1:
                x = pa[x][i]
                if x < 0:
                    return x
        return x

    res = []
    for x1, y1 in queries:
        x, y = x1, y1
        if depth[x] > depth[y]:
            x, y = y, x
        y = get_kth(y, depth[y] - depth[x])
        if x != y:
            for i in range(len(pa[x]) - 1, -1, -1):
                px, py = pa[x][i], pa[y][i]
                if px != py:
                    x, y = px, py
            lca = pa[x][0]
        else:
            lca = x
        max_freq, index = 0, 0
        for i in range(27):
            if max_freq < (f := freq[x1][i] + freq[y1][i] - freq[lca][i] * 2):
                max_freq = f
        res.append((sum(freq[x1]) + sum(freq[y1]) - sum(freq[lca]) * 2) - max_freq)
    return res


def solution_2861(n: int, k: int, budget: int, composition: List[List[int]], stock: List[int], cost: List[int]) -> int:
    ans = 0
    mx = min(stock) + budget  # 假设每个金属只要1个，价格也是1，因此上界是min(stock)+budget

    def check(num: int, comp: List[int]) -> bool:
        money = 0
        for s, base, c in zip(stock, comp, cost):
            if s < base * num:
                money += (base * num - s) * c  # 库存不够
                if money > budget:
                    return False
        return True

    for comp in composition:
        low = ans  # 如果一份都造不出来，那么就还是ans
        high = mx + 1
        while low + 1 < high:
            mid = (low + high) // 2
            if check(mid, comp):
                low = mid
            else:
                high = mid
        ans = low
    return ans


def solution_2808(nums: List[int]) -> int:
    """
    超时
    """
    my_set = set(nums)
    n = len(nums)
    min_cnt = 10 ** 5
    for num in my_set:
        ns_indexes = set()
        num_indexes = set()
        for i in range(len(nums)):
            if nums[i] == num:
                num_indexes.add(i)
            else:
                ns_indexes.add(i)
        cnt = 0
        while True:
            if len(ns_indexes) == 0:
                break
            tmp = []
            for num_index in num_indexes:
                p1, p2 = (num_index + 1) % n, (num_index - 1 + n) % n
                if p1 in ns_indexes:
                    ns_indexes.remove(p1)
                    tmp.append(p1)
                if p2 in ns_indexes:
                    ns_indexes.remove(p2)
                    tmp.append(p2)
            for t in tmp:
                num_indexes.add(t)
            cnt += 1
        min_cnt = min(min_cnt, cnt)
    return min_cnt


def solution_2808_2(nums: List[int]) -> int:
    """
    超时
    """
    my_set = set(nums)
    n = len(nums)
    min_cnt = 10 ** 5
    for num in my_set:
        tmp = []
        for i in range(len(nums)):
            if nums[i] == num:
                tmp.append(i)
        # 记录所有端点
        # 端点(i,j)之间变为x的时间为(j-i//2)
        cnt = 0
        for i in range(len(tmp) - 1):
            cnt = max(cnt, (tmp[i + 1] - tmp[i]) // 2)
        # n-2 到 2 之间为 n-1,0,1需要2次
        # (n-j+i)//2
        cnt = max(cnt, (n - tmp[-1] + tmp[0]) // 2)
        min_cnt = min(min_cnt, cnt)
    return min_cnt


def solution_2808_3(nums: List[int]) -> int:
    """
    和上面没差多少
    一些细节的优化
    """
    mp = defaultdict(list)
    res = n = len(nums)
    for i, a in enumerate(nums):
        mp[a].append(i)
    for pos in mp.values():
        mx = pos[0] + n - pos[-1]
        for i in range(len(pos) - 1):
            mx = max(mx, pos[i + 1] - pos[i])
        res = min(res, mx // 2)
    return res


def solution_2670(nums: List[int]) -> List[int]:
    n = len(nums)
    suf = [0] * (n + 1)  # s[n]用于计算最后一个下标的后缀数量
    s = set()
    for i in range(n - 1, 0, -1):
        s.add(nums[i])
        suf[i] = len(s)
    s.clear()
    ans = [0] * n
    for i, x in enumerate(nums):
        s.add(x)
        ans[i] = len(s) - suf[i + 1]
    return ans


def solution_2641(root: Optional[TreeNode]) -> Optional[TreeNode]:
    s = []

    def dfs1(p, level):
        if not p:
            return
        if len(s) < level:
            s.append(p.val)
        else:
            s[level - 1] += p.val
        dfs1(p.left, level + 1)
        dfs1(p.right, level + 1)

    def dfs2(p, level, q):
        if not p:
            return
        lv = p.left.val if p.left else 0
        rv = p.right.val if p.right else 0
        p.val = s[level - 1] - p.val - q
        dfs2(p.left, level + 1, rv)
        dfs2(p.right, level + 1, lv)

    dfs1(root, 1)
    dfs2(root, 1, 0)
    return root


def solution_2641_2(root: Optional[TreeNode]) -> Optional[TreeNode]:
    # BFS
    root.val = 0
    q = [root]
    while q:
        tmp = q
        q = []
        next_level_sum = 0
        for node in tmp:
            if node.left:
                q.append(node.left)
                next_level_sum += node.left.val
            if node.right:
                q.append(node.right)
                next_level_sum += node.right.val

        for node in tmp:
            child_sum = (node.left.val if node.left else 0) + (node.right.val if node.right else 0)
            if node.left:
                node.left.val = next_level_sum - child_sum
            if node.right:
                node.right.val = next_level_sum - child_sum
    return root


def solution_2583(root: Optional[TreeNode], k: int) -> int:
    res = []
    q = [root]
    while q:
        tmp = q
        q = []
        vals = []
        for node in tmp:
            vals.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(sum(vals))
    if len(res) >= k:
        res.sort(reverse=True)
        return res[k - 1]
    else:
        return -1


def solution_2867(n: int, edges: List[List[int]]) -> int:
    MX = 10 ** 5 + 1
    is_prime = [True] * MX
    is_prime[1] = False
    for i in range(2, isqrt(MX) + 1):
        if is_prime[i]:
            for j in range(i * i, MX, i):
                is_prime[j] = False
    g = [[] for _ in range(n + 1)]
    for x, y in edges:
        g[y].append(x)
        g[x].append(y)

    def dfs(x, fa):
        nodes.append(x)
        for y in g[x]:
            if y != fa and not is_prime[y]:
                dfs(y, x)

    ans = 0
    size = [0] * (n + 1)
    for x in range(1, n + 1):
        if not is_prime[x]:
            continue
        s = 0
        for y in g[x]:  # 质数 x 把这棵树分成了若干个连通块
            if is_prime[y]:
                continue
            if size[y] == 0:  # 尚未计算过
                nodes = []
                dfs(y, -1)  # 遍历 y 所在连通块，在不经过质数的前提下，统计有多少个非质数
                for z in nodes:
                    size[z] = len(nodes)
            # 这 size[y] 个非质数与之前遍历到的 s 个非质数，两两之间的路径只包含质数 x
            ans += size[y] * s
            s += size[y]
        ans += s  # 从 x 出发的路径
    return ans


def solution_2673(n: int, cost: List[int]) -> int:
    ans = 0
    for i in range(n // 2, 0, -1):
        ans += abs(cost[2 * i - 1] - cost[2 * i])
        cost[i - 1] += max(cost[2 * i - 1], cost[2 * i])
    return ans


def solution_2581(edges: List[List[int]], guesses: List[List[int]], k: int) -> int:
    n = len(edges) + 1
    g = [[] for _ in range(n)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)
    cnt = [0] * n
    fa = [-1] * n

    def dfs1(x, p):
        fa[x] = p
        for y in g[x]:
            if y != p:
                dfs1(y, x)

    dfs1(0, -1)
    guesses_set = set()
    for u, v in guesses:
        guesses_set.add((u, v))
        if fa[v] == u:
            cnt[0] += 1
    ans = 0 if cnt[0] < k else 1

    def dfs2(x, p):
        nonlocal ans
        c = cnt[p]
        if (x, p) in guesses_set:
            c += 1
        if (p, x) in guesses_set:
            c -= 1
        if c >= k:
            ans += 1
        cnt[x] = c
        for y in g[x]:
            if y != p:
                dfs2(y, x)

    for t in g[0]:
        dfs2(t, 0)

    return ans


def solution_2917(nums: List[int], k: int) -> int:
    cnt = defaultdict(int)
    for num in nums:
        cur = 0
        while num > 0:
            cnt[cur] += num & 1
            num = num >> 1
            cur += 1
    ans = 0
    for c in cnt:
        if cnt[c] >= k:
            ans += 1 << c
    return ans
