import heapq
from bisect import bisect_left
from itertools import accumulate

from .contest_solution import *
from .data_struct import *


def solution_3014(word: str) -> int:
    return solution_100191(word)


def solution_3016(word: str) -> int:
    return solution_100192(word)


def solution_3016_2(word: str) -> int:
    return solution_100192_2(word)


def solution_3015(n: int, x: int, y: int) -> List[int]:
    return solution_100188(n, x, y)


def solution_3015_2(n: int, x: int, y: int) -> List[int]:
    return solution_100188_2(n, x, y)


def solution_3017(n: int, x: int, y: int) -> List[int]:
    """
    距离 = 两个编号之差
    i-j的距离 = abs(i-j)

    没有x到y的情况下
    差分数组: 把一堆连续的数字都加1
    对于每个i
    i到左边的房子: [1,i-1] -> +=1
    i到右边的房子: [1,n-1] -> +=1


    加上x,y
    1. 1 <= i <= x
        撤销[y-i,n-i] -> +=1

        对于y而言:原本距离y-i变为x-i+1
        距离缩短 (y-i)-(x-i)+1
            dec = y-x-1

        则 [y-i-dec,n-i-dec] -> +=1

        对于 j<y,若j-i>x-i+1+y-j,则距离缩短
            => 2j>x+y+1
        j = (x+y+1)/2+1
        从j到y-1,距离缩短了

        撤销 [j-i,y-1-i] -> +=1

        dec = (j-i)-(x-i+1+y-j) = 2j - (x+y+1)
        则 [x-i+2,x-i+y-j+1] -> +=1

    2. x < i < (x+y)/2
        对于y及其后面的编号
        dec = (y-i) - (i-x+1)
        撤销[y-i,n-i] -> +=1
        [y-i-dec,n-i-dec] -> +=1

        对于j到y=1
        如果j-i>i-x+1+y-j,则距离缩短
        j > i+(y-x+1)/2
        j = i+(y-x+1)/2+1到y-1都可以缩减距离
        dec = (y-i)-(i-x+1+y) = -2i+x-1
        撤销 [j-i,y-1-i] -> +=1
        [i-x+2, i-x+y-j+1] -> +=1

    3. (x+y)/2 < i < y # 通过对称可以回到2

    4. y <= i <= n # 通过对称可以回到1
    """
    if x > y:
        x, y = y, x
    diff = [0] * (n + 1)

    def add(left: int, right: int, value: int) -> None:
        if left > right:
            return
        diff[left] += value
        diff[right + 1] -= value

    def update(i: int, x: int, y: int) -> None:
        add(y - i, n - i, -1)
        dec = y - x - 1
        add(y - i - dec, n - i - dec, 1)
        j = (x + y + 1) // 2 + 1
        add(j - i, y - 1 - i, -1)
        add(x - i + 2, x - i + y - j + 1, 1)

    def update2(i: int, x: int, y: int) -> None:
        add(y - i, n - i, -1)
        dec = y - 2 * i + x - 1
        add(y - i - dec, n - i - dec, 1)
        j = i + (y - x + 1) // 2 + 1
        add(j - i, y - 1 - i, -1)
        add(i - x + 2, i - x + y - j + 1, 1)

    for i in range(1, n + 1):
        add(1, i - 1, 1)
        add(1, n - i, 1)
        if x + 1 >= y:
            continue
        if i <= x:
            update(i, x, y)
        elif i >= y:
            update(n + 1 - i, n + 1 - y, n + 1 - x)
        elif i < (x + y) // 2:
            update2(i, x, y)
        elif i > (x + y + 1) // 2:
            update2(n + 1 - i, n + 1 - y, n + 1 - x)
    return list(accumulate(diff))[1:]


def solution_3017_2(n: int, x: int, y: int) -> List[int]:
    """
    思路参考题解的直接计算部分:
    https://leetcode.cn/problems/count-the-number-of-houses-at-a-certain-distance-ii/solutions/2613373/yong-che-xiao-de-fang-shi-si-kao-pythonj-o253/
    """
    if x > y:
        x, y = y, x

    if x + 1 >= y:
        return list(range((n - 1) * 2, -1, -2))

    diff = [0] * (n + 1)

    def add(l: int, r: int) -> None:
        diff[l] += 2
        diff[r + 1] -= 2

    for i in range(1, n):
        if i <= x:
            k = (x + y + 1) // 2
            add(1, k - i)
            add(x - i + 2, x - i + y - k)
            add(x - i + 1, x - i + 1 + n - y)
        elif i < (x + y) // 2:
            k = i + (y - x + 1) // 2
            add(1, k - i)
            add(i - x + 2, i - x + y - k)
            add(i - x + 1, i - x + 1 + n - y)
        else:
            add(1, n - i)

    return list(accumulate(diff))[1:]


def solution_3019(s: str) -> int:
    return solution_100215(s)


def solution_3020(nums: List[int]) -> int:
    """
    慢
    """
    return solution_100206(nums)


def solution_3020_2(nums: List[int]) -> int:
    cnt = Counter[int](nums)
    ans = cnt[1] - (cnt[1] % 2 ^ 1)  # 没有1该值为-1 = 0-1
    del cnt[1]
    for x in cnt:
        res = 0
        while True:
            if x not in cnt:
                res -= 1
                break
            if cnt[x] == 1:
                res += 1
                break
            res += 2
            x *= x
        ans = max(ans, res)
    return ans


def solution_3021(n: int, m: int) -> int:
    return solution_100195(n, m)


def solution_3022(nums: List[int], k: int) -> int:
    return solution_100179(nums, k)


def solution_3024(nums: List[int]) -> str:
    return solution_100222(nums)


def solution_3024_2(nums: List[int]) -> str:
    """
    简洁一点
    """
    nums.sort()
    x, y, z = nums
    if x + y <= z:  # y+z一定大于等于x,x+z一定大于等于y
        return 'none'
    if x == z:
        return 'equilateral'
    if x == y or y == z:
        return 'isosceles'
    return 'scalene'


def solution_3025(points: List[List[int]]) -> int:
    return solution_100194(points)


def solution_3026(nums: List[int], k: int) -> int:
    """
    超时
    """
    return solution_100183(nums, k)


def solution_3026_2(nums: List[int], k: int) -> int:
    """
    前缀和与哈希表

    满足 |nums[i] - nums[j]|==k
    计算 s[j+1]-s[i]的最大值
    枚举 j, 问题变成计算s[i]的最小值

    维护哈希表,key 是 nums[i], value是s[i]的最小值
    """
    # 前缀和
    ans = -inf
    s = 0
    f = defaultdict(lambda: inf)
    for x in nums:
        # 维护前缀和的最小值,不包含x
        f[x] = min(f[x], s)
        s += x
        # 在遍历中找符合条件的nums[j]和nums[i]
        # 这样保证这里出现的有效的x-k和x+k的下标一定比x小
        # 满足构成子数组的要求
        ans = max(ans, s - f[x - k], s - f[x + k])
    return ans if ans > -inf else 0


def solution_3027(points: List[List[int]]) -> int:
    return solution_100193(points)


def solution_3028(nums: List[int]) -> int:
    return solution_100214(nums)


def solution_3029(word: str, k: int) -> int:
    return solution_100204(word, k)


def solution_3030(image: List[List[int]], threshold: int) -> List[List[int]]:
    return solution_100189(image, threshold)


def solution_3031(word: str, k: int) -> int:
    return solution_100204(word, k)


def solution_3031_2(word: str, k: int) -> int:
    """
    z函数(扩展KMP)
    """
    n = len(word)
    z = [0] * n
    left = right = 0  # z-box范围
    for i in range(1, n):
        if i <= right:  # i在z-box内
            z[i] = min(z[i - left], right - i + 1)
        # 向后暴力匹配
        while i + z[i] < n and word[z[i]] == word[i + z[i]]:
            left, right = i, i + z[i]
            z[i] += 1
        # 后缀完全匹配前缀
        if i % k == 0 and z[i] >= n - i:
            return i // k
    return (n - 1) // k + 1


def solution_3033(matrix: List[List[int]]) -> List[List[int]]:
    return solution_100230(matrix)


def solution_3034(nums: List[int], pattern: List[int]) -> int:
    return solution_100186(nums, pattern)


def solution_3035(words: List[str]) -> int:
    return solution_100219(words)


def solution_3036(nums: List[int], pattern: List[int]) -> int:
    return solution_100198(nums, pattern)


def solution_3039(s: str) -> str:
    return solution_100211(s)


def solution_3039_2(s: str) -> str:
    """
    1. 不会有重复字母
    2. 出现次数最多的字母
    3. 最后一次出现的下标
    """
    last = {c: i for i, c in enumerate(s)}
    cnt = Counter(s)
    mx = max(cnt.values())
    ids = sorted(last[ch] for ch, c in cnt.items() if c == mx)
    return ''.join(s[i] for i in ids)


def solution_3038(nums: List[int]) -> int:
    return solution_100221(nums)


def solution_3040(nums: List[int]) -> int:
    return solution_100220(nums)


def solution_3040_2(nums: List[int]) -> int:
    """
    子问题是从两侧向内缩小的 -> 区间DP
    dfs(i, j) = 操作 a[i] ... a[j] 这一段子数组(闭区间[i, j])的最多可以进行的操作字数
    O(n^2)
    """

    @cache
    def dfs(i, j, target):
        if i >= j:
            return 0
        res = 0
        if nums[i] + nums[i + 1] == target:
            res = max(res, dfs(i + 2, j, target) + 1)
        if nums[j - 1] + nums[j] == target:
            res = max(res, dfs(i, j - 2, target) + 1)
        if nums[i] + nums[j] == target:
            res = max(res, dfs(i + 1, j - 1, target) + 1)
        return res

    n = len(nums)
    res1 = dfs(2, n - 1, nums[0] + nums[1])
    res2 = dfs(0, n - 3, nums[-1] + nums[-2])
    res3 = dfs(1, n - 2, nums[0] + nums[-1])
    return max(res1, res2, res3) + 1


def solution_3041(nums: List[int]) -> int:
    """
    排序的正确性
    选出来的排序后的序列是 b
    b[i] +1 = b[i+1] -> 操作前, 不会出现b[i] > b[i+1]
    """
    return solution_100205(nums)


def solution_3041_2(nums: List[int]) -> int:
    """
    子序列DP
    1. 01背包: 子序列相邻元素无关
    2. 最长递增子序列 LIS: 子序列相邻元素相关
        2.1 O(n^2) (f[i] nums[i]) 找 j<i 且 nums[j] < nums[i], 从这样的f[j] 转移过来取max

    如果不考虑nums变化
    dfs(i) = dfs(j)+1:
        满足 nums[i] = nums[j]+1, 二分找右边第一个nums[i]+1
        从而O(log n)时间找到转移来源

    考虑nums是否加1
    dfs(i,add_one) = max(dfs(j,0), dfs(j,1))+1
        满足 nums[i] = nums[j]+ add_one 或者 num[i] = nums[j] + add_one + 1
        分别递归到 dfs(j,1) 和 dfs(j,0)
    """

    def bisearch(l, r, x):
        while l + 1 < r:
            mid = (l + r) // 2
            if nums[mid] > x:
                r = mid
            else:
                l = mid
        if l == -1 or nums[l] < x:
            return -1
        return l

    n = len(nums)
    if n == 1:
        return 1
    nums.sort()
    dp = [[0] * 2 for _ in range(n)]
    dp[0][0], dp[0][1] = 1, 1
    ans = 1
    for i in range(1, n):
        dp[i][0], dp[i][1] = 1, 1
        right = i
        cur = nums[i]
        while (j := bisearch(-1, right, cur)) != -1:
            dp[i][1] = max(dp[i][1], dp[j][0] + 1)
            right = j
        while (j := bisearch(-1, right, cur - 1)) != -1:
            dp[i][0] = max(dp[i][0], dp[j][0] + 1)
            dp[i][1] = max(dp[i][1], dp[j][1] + 1)
            right = j
        while (j := bisearch(-1, right, cur - 2)) != -1:
            dp[i][0] = max(dp[i][0], dp[j][1] + 1)
            right = j
        ans = max(dp[i][0], dp[i][1], ans)
    return ans


def solution_3042(words: List[str]) -> int:
    return solution_100212(words)


def solution_3043(arr1: List[int], arr2: List[int]) -> int:
    st = set()
    for s in map(str, arr1):
        for i in range(1, len(s) + 1):
            st.add(s[:i])
    ans = 0
    for s in map(str, arr2):
        for i in range(1, len(s) + 1):
            if s[:i] not in st:
                break
            ans = max(ans, i)
    return ans


def solution_3044(mat: List[List[int]]) -> int:
    return solution_100217(mat)


def solution_3045(words: List[str]) -> int:
    """
    字典树

    1. 把字符串按照前缀分组
    2. 用树实现
    3. 本题做法
    把 s 转成一个pair列表 [(s[0],s[n-1]),...,(s[n-1],s[0])]
    判断 words[i] 对应的 pair 列表是不是word[j]对应的 pair 列表的前缀
    """
    ans = 0
    root = TireOf3045()
    for s in words:
        tmp = list(zip(s, s[::-1]))
        i = 0
        p = root
        while i < len(tmp):
            ans += p.cnt
            pair = tmp[i]
            if pair not in p.son:
                p.son[pair] = TireOf3045()
            p = p.son[pair]
            i += 1
        ans += p.cnt
        p.cnt += 1
    return ans


def solution_3046(nums: List[int]) -> bool:
    return weekly_contest_386_solution_1(nums)


def solution_3047(bottomLeft: List[List[int]], topRight: List[List[int]]) -> int:
    """
    交集面积：
    1. 左下角坐标：两个正方形左下角坐标的横纵最大值 max(x00,x10),max(y00,y10)
    2. 右上角坐标：两个正方形左下角坐标的横纵最小值 min(x01,x11),min(y01,y11)
    """
    ans = 0
    for i in range(len(bottomLeft)):
        for j in range(i + 1, len(bottomLeft)):
            x0 = max(bottomLeft[i][0], bottomLeft[j][0])
            y0 = max(bottomLeft[i][1], bottomLeft[j][1])
            x1 = min(topRight[i][0], topRight[j][0])
            y1 = min(topRight[i][1], topRight[j][1])
            size = min(x1 - x0, y1 - y0)
            if size > 0:
                ans = max(ans, size * size)
    return ans


def solution_3048(nums: List[int], changeIndices: List[int]) -> int:
    """
    已知答案去判断能否成立，相比直接求答案更容易时： 二分答案
    需保证能够二分答案
    对于本题：
    答案越大，越能够搞定所有数字变为0并验证的可能性
    -> 有单调性,可以二分答案

    验证时间越晚越好，用来变0的时间越多
    """

    # 考虑前mx天
    def check(mx: int):
        last_t = [-1] * n
        for t, idx in enumerate(changeIndices[:mx]):
            last_t[idx - 1] = t
        if -1 in last_t:
            return False
        cnt = 0
        for i, idx in enumerate(changeIndices[:mx]):
            idx -= 1
            if i == last_t[idx]:  # 一定要标记，后面没机会了
                if nums[idx] > cnt:
                    return False
                cnt -= nums[idx]
            else:
                cnt += 1
        return True

    n = len(nums)
    m = len(changeIndices)
    left = n + sum(nums) - 1
    right = m + 1
    while left + 1 < right:
        mid = (left + right) // 2
        if check(mid):
            right = mid
        else:
            left = mid
    return right if right < m + 1 else -1
    # left = n + sum(nums)
    # # bisect_left 返回值是range中的idx
    # ans = left + bisect_left(range(left, m + 1), True, key=check)
    # return -1 if ans > m else ans


def solution_3049(nums: List[int], changeIndices: List[int]) -> int:
    """
    已知答案去判断能否成立，相比直接求答案更容易时： 二分答案
    需保证能够二分答案
    对于本题：
    答案越大，越能够搞定所有数字变为0并验证的可能性
    -> 有单调性,可以二分答案

    减一   (慢速复习) -> 随意复习
    置零   (快速复习) -> 涉及到changeIndices
    标记   (考试)    -> 参加任意一门课程的考试 -> 排的越靠后越好

    先慢速复习，再快速复习，那么前面的慢速复习做别的事情更好

    ** 对于一门课程，要么全部用慢速复习，要么在某一天快速复习

    倒着遍历，如果这一天不是快速复习，那么和第三题一样 cnt+=1
    如果遇到快速复习的那天
    1. 执行快速复习 需消耗1天进行考试
    2. 不快速复习
        1. nums[i] = 0
        2. nums[i] = 1
        3. cnt = 0 无法快速复习，没时间拿来考试了
            -> 不一定只能提早慢速复习
            -> 反悔贪心
            -> 取一个原本用快速复习搞定的，且nums[i]最小的，反悔这门课程的用时(快速复习的一天+考试的一天)
            -> 多出来的这 2 天时间，用来做当前这门课程的快速复习+考试
    从右往左遍历结束后，检查没有考试的课程，用慢速复习搞定
    """
    n = len(nums)
    m = len(changeIndices)
    first_t = [-1] * n
    for t in range(m - 1, -1, -1):
        first_t[changeIndices[t] - 1] = t

    def check(mx: int) -> bool:
        cnt = 0
        done = [False] * n  # 可以直接统计天数
        h = []
        for t in range(mx - 1, -1, -1):
            i = changeIndices[t] - 1
            v = nums[i]
            if v <= 1 or first_t[i] != t:
                cnt += 1  # 留着给前面，用来-1或者标记
                continue
            if cnt == 0:
                # 没办法反悔,现在没有快速复习的科目或者花时间最少的科目也比现在的科目大
                if not h or v <= h[0][0]:
                    cnt += 1  # 留着给前面，用来-1或者标记
                    continue
                # 反悔
                done[heapq.heappop(h)[1]] = False
                cnt += 2
            # 入堆
            done[i] = True
            cnt -= 1  # t这天nums[i]置0 cnt中一天用于标记
            heapq.heappush(h, (v, i))

        for i, b in enumerate(done):
            if not b:
                cnt -= nums[i] + 1  # 慢速复习+考试
        return cnt >= 0

    total = n + sum(nums)

    def check2(mx: int) -> bool:
        cnt = 0
        slow = total
        h = []
        for t in range(mx - 1, -1, -1):
            i = changeIndices[t] - 1
            v = nums[i]
            if v <= 1 or first_t[i] != t:
                cnt += 1  # 留着给前面，用来-1或者标记
                continue
            if cnt == 0:
                # 没办法反悔,现在没有快速复习的科目或者花时间最少的科目也比现在的科目大
                if not h or v <= h[0]:
                    cnt += 1  # 留着给前面，用来-1或者标记
                    continue
                # 反悔
                slow += heapq.heappop(h) + 1
                cnt += 2
            slow -= v + 1
            cnt -= 1
            heapq.heappush(h, v)
        return cnt >= slow

    # left = n - 1
    # right = m + 1
    # while left + 1 < right:
    #     mid = (left + right) // 2
    #     if check(mid):
    #         right = mid
    #     else:
    #         left = mid
    # return right if right < m + 1 else -1
    ans = n + bisect_left(range(n, m + 1), True, key=check2)
    return -1 if ans > m else ans


def solution_3065(nums: List[int], k: int) -> int:
    return biweekly_contest_125_solution_1(nums, k)


def solution_3066(nums: List[int], k: int) -> int:
    return biweekly_contest_125_solution_2(nums, k)


def solution_3067(edges: List[List[int]], signalSpeed: int) -> List[int]:
    # TL
    return biweekly_contest_125_solution_3(edges, signalSpeed)


def solution_3067_2(edges: List[List[int]], signalSpeed: int) -> List[int]:
    n = len(edges) + 1
    g = [[] for _ in range(n)]
    for x, y, w in edges:
        g[x].append((y, w))
        g[y].append((x, w))

    def dfs(x, fa, path_sum):
        cnt = 0 if path_sum % signalSpeed else 1
        for y, wt in g[x]:
            if y != fa:
                cnt += dfs(y, x, path_sum + wt)
        return cnt

    ans = [0] * n
    for i, gi in enumerate(g):
        s = 0
        for y, w in gi:
            cnt = dfs(y, i, w)
            ans[i] += cnt * s
            s += cnt
    return ans


def solution_3068(nums: List[int], k: int, edges: List[List[int]]) -> int:
    """
    1. 对于一条路径上的边都操作一次
    => 1-2-3-4
    => 1 0 0 1
    => 只把路径的起点和终点异或了K, 其余中间节点不变

    2. 被操作的两个数, 可以分为哪些情况?
        两个数都没有异或K => 多了两个异或K的数
        两个数都异或了K   => 少了两个异或K的数
        1个异或K, 1个没有 => 总和不变
        => 无论操作多少次,总有偶数个数异或了K

    3. 问题变成 从nums中选偶数个数,异或K,得到的最大元素和是多少
    4. 每个数字独立考虑, 是否异或K DP
    """
    n = len(nums)
    f = [[0, 0] for _ in range(n)]
    f[0][0] = nums[0]
    f[0][1] = nums[0] ^ k
    for i, num in enumerate(nums):
        if i == 0:
            continue
        f[i][0] = max(f[i - 1][0] + num, f[i - 1][1] + (num ^ k))
        f[i][1] = max(f[i - 1][0] + (num ^ k), f[i - 1][1] + num)
    return f[-1][0]


def solution_3068_2(nums: List[int], k: int, edges: List[List[int]]) -> int:
    """
    树形DP
    """
    g = [[] for _ in nums]
    for x, y in edges:
        g[x].append(y)
        g[y].append(x)

    def dfs(x, fa):
        f0, f1 = 0, -inf
        for y in g[x]:
            if y != fa:
                r0, r1 = dfs(y, x)
                f0, f1 = max(f0 + r0, f1 + r1), max(f1 + r0, f0 + r1)
        return max(f0 + nums[x], f1 + (nums[x] ^ k)), max(f1 + nums[x], f0 + (nums[x] ^ k))

    return dfs(0, -1)[0]


def solution_3068_3(nums: List[int], k: int, edges: List[List[int]]) -> int:
    """
    贪心1
    """
    s = sum(nums)
    a = []
    b = []
    for x in nums:
        d = (x ^ k) - x
        if d > 0:
            a.append(d)  # 变大的
        else:
            b.append(d)
    sa = sum(a)
    if len(a) % 2 == 0:
        return s + sa
    else:
        res = s + sa - min(a)
        if b and res < (tmp := s + sa + max(b)):
            res = tmp
        return res


def solution_3068_4(nums: List[int], k: int, edges: List[List[int]]) -> int:
    """
    贪心2
    排序找异或使得结果变大的最大两个值
    """
    li = []
    c = 0
    for v in nums:
        c += v
        li.append((v ^ k) - v)
    li.sort()
    for i in range(len(li) - 1, 0, -2):
        x = li[i] + li[i - 1]
        if x > 0:  # 异或后变大了
            c += x
        else:
            break
    return c


def solution_3069(nums: List[int]) -> List[int]:
    return weekly_contest_387_solution_1(nums)


def solution_3070(grid: List[List[int]], k: int) -> int:
    return weekly_contest_387_solution_2(grid, k)


def solution_3071(grid: List[List[int]]) -> int:
    return weekly_contest_387_solution_3(grid)


def solution_3072(nums: List[int]) -> List[int]:
    # 树状数组
    sorted_nums = sorted(set(nums))
    m = len(sorted_nums)
    a = [nums[0]]
    b = [nums[1]]
    t1 = Fenwick(m + 1)
    t2 = Fenwick(m + 1)
    t1.add(bisect_left(sorted_nums, nums[0]) + 1, 1)
    t2.add(bisect_left(sorted_nums, nums[1]) + 1, 1)
    for x in nums[2:]:
        v = bisect_left(sorted_nums, x) + 1
        gc1 = len(a) - t1.pre(v)  # a中元素个数 减去 现在t1中已有的比v小的元素个数
        gc2 = len(b) - t2.pre(v)
        if gc1 > gc2 or gc1 == gc2 and len(a) <= len(b):
            a.append(x)
            t1.add(v, 1)
        else:
            b.append(x)
            t2.add(v, 1)
    return a + b


def solution_3074(apple: List[int], capacity: List[int]) -> int:
    return weekly_contest_388_solution_1(apple, capacity)


def solution_3075(happiness: List[int], k: int) -> int:
    return weekly_contest_388_solution_2(happiness, k)


def solution_3076(arr: List[str]) -> List[str]:
    return weekly_contest_388_solution_3(arr)


def solution_3077(nums: List[int], k: int) -> int:
    return weekly_contest_388_solution_4(nums, k)
