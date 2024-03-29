import heapq
from collections import defaultdict
from functools import cache
from itertools import pairwise
from math import inf
from typing import *


def solution_100191(word: str) -> int:
    """
    index : 3014
    """
    if len(word) <= 8:
        return len(word)
    if len(word) <= 16:
        return (len(word) - 8) * 2 + 8
    if len(word) <= 24:
        return (len(word) - 16) * 3 + 24
    return (len(word) - 24) * 4 + 48


def solution_100188(n: int, x: int, y: int) -> List[int]:
    """
    index:3015
    """
    # floyd
    # 超时
    f = [[10 ** 18] * n for i in range(n)]
    f[x - 1][y - 1], f[y - 1][x - 1] = 1, 1
    for i in range(n):
        f[i][i] = 0
    for i in range(n - 1):
        f[i][i + 1], f[i + 1][i] = 1, 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                f[i][j] = min(f[i][j], (f[i][k] + f[k][j]))
    res = [0] * n
    for i in range(n):
        for j in range(n):
            res[f[i][j] - 1] += 1
    res[-1] = 0
    return res


def solution_100188_2(n: int, x: int, y: int) -> List[int]:
    # 暴力做法:BFS
    # 每个点花费O(n)求出它到其余点的距离
    # 花费O(n^2)时间求出所有结果
    res = [0] * n
    x = x - 1
    y = y - 1
    # 只往后找,因此每次结果加+2
    for i in range(n):
        for j in range(i + 1, n):
            d1 = j - i  # 直接走
            d2 = abs(i - x) + 1 + abs(j - y)  # i->x->y->j
            d3 = abs(i - y) + 1 + abs(j - x)  # i->y->x->j
            min_d = min(d1, d2, d3)
            res[min_d - 1] += 2
    return res


def solution_100192(word: str) -> int:
    """
    index: 3016
    """
    f = [0] * 26
    for ch in word:
        f[ord(ch) - ord('a')] += 1
    f.sort(reverse=True)
    res = 0
    res += sum(f[0:8])
    res += sum(f[8:16] * 2)
    res += sum(f[16:24] * 3)
    res += sum(f[24:] * 4)
    return res


def solution_100192_2(word: str) -> int:
    # 排序不等式
    cnt = Counter[str](word)
    a = sorted(cnt.values(), reverse=True)
    ans = 0
    for i, c in enumerate(a):
        ans += c * (i // 8 + 1)
    return ans


def solution_100215(s: str) -> int:
    tmp = s.lower()
    cnt = 0
    for i in range(len(tmp) - 1):
        if tmp[i] != tmp[i + 1]:
            cnt += 1
    return cnt


def solution_100206(nums: List[int]) -> int:
    f = {}
    for num in nums:
        if num not in f:
            f[num] = 1
        else:
            f[num] += 1
    max_cnt = 1
    # 从f中拿就可以了
    for num in nums:
        cnt = 2
        # cnt = f[num]- (f[num]%2^1)
        if num == 1 and f[num] % 2 == 0:
            cnt = f[num] - 1
        elif num == 1 and f[num] % 2 != 0:
            cnt = f[num]
        elif f[num] >= 2:
            tmp = num
            while True:
                tmp = tmp * tmp
                if tmp not in f:
                    cnt -= 1
                    break
                elif f[tmp] == 1:
                    cnt += 1
                    break
                elif f[tmp] >= 2:
                    cnt += 2
        else:
            cnt -= 1
        max_cnt = max(max_cnt, cnt)
    return max_cnt


def solution_100195(n: int, m: int) -> int:
    even_x = (n + 1) // 2
    even_y = (m + 1) // 2
    odd_x = n - even_x
    odd_y = m - even_y
    return even_x * odd_y + even_y * odd_x
    # return n*m//2


def solution_100179(nums: List[int], k: int) -> int:
    """
    1. 拆位
    2. 试填法
    3. 相邻合并 -> 连续子数组合并

    从左到右考虑数字
    如果某一段数字能合并成0，那么操作次数就是这一段的长度-1
    如果某一段数字不能合并成0，那么操作次数就是这一段的长度（从其他地方借0）

    考虑每一位的时候，需要带上高位可以变成0的位
    """
    ans = mask = 0  # mask表示需要考虑的位置
    for b in range(29, -1, -1):
        mask |= 1 << b  # 当前考虑的位
        cnt = 0
        and_res = -1  # 全为1的数字
        for x in nums:
            and_res &= x & mask
            if and_res:
                cnt += 1  # 还得合并
            else:  # 这一段合并完了
                and_res = -1  # 合并下一段
        # 题目条件 k<len(nums), 故cnt > k时视为无法变0
        if cnt > k:
            ans |= 1 << b
            mask ^= 1 << b  # 反悔，这一位不要了，因为他只能是1
    return ans


def solution_100222(nums: List[int]) -> str:
    if not nums[0] + nums[1] > nums[2]:
        return 'none'
    if not nums[0] + nums[2] > nums[1]:
        return 'none'
    if not nums[1] + nums[2] > nums[0]:
        return 'none'
    ms = set()
    for num in nums:
        ms.add(num)
    if len(ms) == 1:
        return 'equilateral'
    if len(ms) == 2:
        return 'isosceles'
    if len(ms) == 3:
        return 'scalene'


def solution_100194(points: List[List[int]]) -> int:
    cnt = 0
    for point1 in points:
        for point2 in points:
            if point1 == point2:
                continue
            if point1[0] <= point2[0] and point1[1] >= point2[1]:
                flag = False
                for point3 in points:
                    if point3 == point2 or point3 == point1:
                        continue
                    if point1[0] <= point3[0] <= point2[0] and point1[1] >= point3[1] >= point2[1]:
                        flag = True
                        break
                if flag:
                    continue
                else:
                    cnt += 1
    return cnt


def solution_100183(nums: List[int], k: int) -> int:
    """
    超时
    """
    s = [0]
    for i in range(1, len(nums) + 1):
        s.append(s[i - 1] + nums[i - 1])
    f = {}
    for i, x in enumerate(nums):
        if x not in f:
            f[x] = [i]
        else:
            f[x].append(i)
    max_sum = -10 ** 19
    for i, x in enumerate(nums):
        tmp = x + k
        if tmp in f:
            for index in f[tmp]:
                if index > i:
                    max_sum = max(max_sum, s[index + 1] - s[i])
        tmp = x - k
        if tmp in f:
            for index in f[tmp]:
                if index > i:
                    max_sum = max(max_sum, s[index + 1] - s[i])
    return max_sum if max_sum > -10 ** 19 else 0


def solution_100193(points: List[List[int]]) -> int:
    # 排序, 保证按照横坐标从小到大排序，后续枚举只需要考虑纵坐标
    # 横坐标相同时，按纵坐标从大到小排序
    points.sort(key=lambda p: (p[0], -p[1]))
    ans = 0
    for i, (_, y0) in enumerate(points):
        max_y = -inf
        for (_, y1) in points[i + 1:]:
            if max_y < y1 <= y0:
                max_y = y1
                ans += 1
    return ans


def solution_100214(nums: List[int]) -> int:
    s = 0
    cnt = 0
    for num in nums:
        s += num
        if s == 0:
            cnt += 1
    return cnt


def solution_100204(word: str, k: int) -> int:
    n = len(word)
    cnt = (n + k - 1) // k
    for i in range(k, n, k):
        length = n - i
        if word[0:length] == word[i:i + length]:
            cnt = min(i // k, cnt)
    return cnt


def solution_100189(image: List[List[int]], threshold: int) -> List[List[int]]:
    def check(center):
        x, y = center[0], center[1]
        for i in range(x - 1, x + 2, 1):
            if abs(image[i][y - 1] - image[i][y]) > threshold or abs(image[i][y] - image[i][y + 1]) > threshold:
                return False
        for i in range(y - 1, y + 2, 1):
            if abs(image[x - 1][i] - image[x][i]) > threshold or abs(image[x][i] - image[x + 1][i]) > threshold:
                return False
        return True

    m = len(image)
    n = len(image[0])
    result = [[-1] * n for _ in range(m)]
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            center = (i, j)
            if check(center):
                result[i][j] = (sum(image[i - 1][j - 1:j + 2]) + sum(image[i][j - 1:j + 2]) + sum(
                    image[i + 1][j - 1:j + 2])) // 9
    cnt = [[0] * n for _ in range(m)]
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if result[i][j] != -1:
                x, y = i, j
                for k in range(x - 1, x + 2):
                    for s in range(y - 1, y + 2):
                        if cnt[k][s] > 0:
                            image[k][s] = image[k][s] + result[i][j]
                        else:
                            image[k][s] = result[i][j]
                        cnt[k][s] += 1
    for i in range(m):
        for j in range(n):
            if cnt[i][j] > 0:
                image[i][j] = image[i][j] // cnt[i][j]

    return image


def solution_100203(word: str, k: int) -> int:
    return solution_100204(word, k)


def solution_100230(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    m = len(matrix[0])
    ans = [[-1] * m for _ in range(n)]
    tmp = set()
    for j in range(m):
        tmp.clear()
        max_c = -1
        for i in range(n):
            if matrix[i][j] != -1:
                ans[i][j] = matrix[i][j]
                max_c = max(max_c, matrix[i][j])
            else:
                tmp.add(i)
        for index in tmp:
            ans[index][j] = max_c
    return ans


def solution_100186(nums: List[int], pattern: List[int]) -> int:
    n = len(nums)
    m = len(pattern)
    cnt = 0
    for i in range(n - m):
        flag = True
        for j in range(m):
            if pattern[j] == 1:
                if nums[i + j + 1] <= nums[i + j]:
                    flag = False
                    break
            elif pattern[j] == 0:
                if nums[i + j + 1] != nums[i + j]:
                    flag = False
                    break
            elif pattern[j] == -1:
                if nums[i + j + 1] >= nums[i + j]:
                    flag = False
                    break
        if flag:
            cnt += 1
    return cnt


def solution_100219(words: List[str]) -> int:
    """
    奇回文串的特点：偶回文串的特点 + 正中间任意填 -> 最后再填

    优先填短的

    只考虑左半边怎么填
    """
    # n = len(words)
    # arrays = [0] * 26
    # lens = [len(x) for x in words]
    # lens.sort()
    # cnt = 0
    # for word in words:
    #     for char in word:
    #         arrays[ord(char) - ord('a')] += 1
    # for length in lens:
    #     flag = False
    #     if length % 2 == 0:
    #         for i in range(26):
    #             if arrays[i] >= length:
    #                 arrays[i] -= length
    #                 length = 0
    #                 flag = True
    #             if arrays[i] % 2 == 0:
    #                 length -= arrays[i]
    #                 arrays[i] = 0
    #             else:
    #                 length -= (arrays[i] - 1)
    #                 arrays[i] = 1
    #
    #     if flag:
    #         cnt += 1
    cnt = Counter[int]()
    for word in words:
        cnt += Counter[int](word)

    # 计算可用来组成回文串左右两侧的字母个数
    # 只需统计一侧可用字母个数
    left = sum(c // 2 for c in cnt.values())
    # 按照字符串长度，从小到大填入字母
    ans = 0
    words.sort(key=len)
    for word in words:
        m = len(word) // 2
        if left < m:
            break
        left -= m  # 拿出m个字符放入word
        ans += 1
    # 不用管奇数的情况
    # 最后留着没用的字符可以随便填上去
    return ans


def solution_100198(nums: List[int], pattern: List[int]) -> int:
    """
    KMP
    """
    n = len(nums)
    m = len(pattern)
    cnt = 0

    def compute_next(p):
        ret = [0] * m
        ret[0] = -1
        j = -1
        for k in range(1, m):
            while j > -1 and pattern[j + 1] != pattern[k]:
                j = ret[j]
            if p[j + 1] == pattern[k]:
                j += 1
            ret[k] = j
        return ret

    pat_next = compute_next(pattern)
    q = -1
    tmp = []
    for i in range(n - 1):
        cur = nums[i + 1] - nums[i]
        if cur > 0:
            tmp.append(1)
        elif cur == 0:
            tmp.append(0)
        else:
            tmp.append(-1)
    for i in range(n - 1):
        while q > -1 and pattern[q + 1] != tmp[i]:
            q = pat_next[q]
        if pattern[q + 1] == tmp[i]:
            q += 1
        if q == m - 1:
            cnt += 1
            q = pat_next[q]

    return cnt


def solution_100198_2(nums: List[int], pattern: List[int]) -> int:
    """
    Z 函数

    把 pattern 拼在b前面
    中间插入间隔符

    根据Z函数的定义，只要z[i] = m,就找到了1个匹配
    """
    m = len(pattern)
    pattern.append(2)
    pattern.extend((y > x) - (y < x) for x, y in pairwise(nums))

    n = len(pattern)
    z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            z[i] = min(z[i - l], r - i + 1)
        while i + z[i] < n and pattern[z[i]] == pattern[i + z[i]]:
            l, r = i, i + z[i]
            z[i] += 1
    return sum(lcp == m for lcp in z[m + 1:])


def solution_100221(nums: List[int]) -> int:
    if len(nums) <= 2:
        return 1
    cnt = 1
    score = nums[0] + nums[1]
    for i in range(2, len(nums) - 1, 2):
        if nums[i] + nums[i + 1] == score:
            cnt += 1
        else:
            return cnt
    return cnt


def solution_100211(s: str) -> str:
    f = defaultdict(lambda: 0)
    idx = {}
    for i in range(len(s)):
        f[s[i]] += 1
        idx[s[i]] = i
    max_cnt = 0
    for ch in f:
        if f[ch] > max_cnt:
            max_cnt = f[ch]
    res = []
    for ch in f:
        if f[ch] == max_cnt:
            res.append((idx[ch], ch))
    res.sort()
    ans = ''
    for _, ch in res:
        ans += ch
    return ans


def solution_100220(nums: List[int]) -> int:
    if len(nums) <= 2:
        return 1

    @cache
    def dfs(i, j, score, cnt):
        if i >= j:
            return cnt
        if nums[i] + nums[i + 1] == score:
            ans1 = dfs(i + 2, j, score, cnt + 1)
        else:
            ans1 = cnt
        if nums[j - 1] + nums[j] == score:
            ans2 = dfs(i, j - 2, score, cnt + 1)
        else:
            ans2 = cnt
        if nums[i] + nums[j] == score:
            ans3 = dfs(i + 1, j - 1, score, cnt + 1)
        else:
            ans3 = cnt
        return max(ans1, ans2, ans3)

    return max(dfs(2, len(nums) - 1, nums[0] + nums[1], 1),
               dfs(0, len(nums) - 3, nums[-2] + nums[-1], 1),
               dfs(1, len(nums) - 2, nums[0] + nums[-1], 1))


def solution_100205(nums: List[int]) -> int:
    # nums.sort()
    # inc = -1
    # cur = nums[0]
    # index = 0
    # w = -inf
    # i = 0
    # while i < len(nums) - 1:
    #     i += 1
    #     cur += 1
    #     if nums[i] == cur:
    #         continue
    #     elif nums[i] - cur == -1 and inc < 0:
    #         inc = i
    #     elif nums[i] - cur == -1 and inc >= 0:
    #         inc = i
    #     elif nums[i] - cur == -2 and inc < 0:
    #         inc = i
    #     else:
    #         w = max(i - index, w)
    #         if inc >= 0:
    #             i = inc
    #         cur = nums[i]
    #         index = i
    #         inc = -1
    # w = max(len(nums) - index, w)
    # return w
    nums.sort()
    f = defaultdict(int)
    for num in nums:
        f[num + 1] = f[num] + 1  # 当前数加1, 可以使得上个为num的数后续增加一个数字
        # 先算num+1, 否则会重复计算自身
        f[num] = f[num - 1] + 1  # 当前数不加1, 可以使得上个为num-1的数后续增加一个数字
    return max(f.values())


def solution_100212(words: List[str]) -> int:
    def isPrefixAndSuffix(str1, str2):
        if len(str1) > len(str2):
            return False
        n = len(str1)
        m = len(str2)
        if str1 != str2[0:n]:
            return False
        if str1 != str2[m - n:]:
            return False
        return True

    cnt = 0
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            if isPrefixAndSuffix(words[i], words[j]):
                cnt += 1
    return cnt


def solution_100217(mat: List[List[int]]) -> int:
    m = len(mat)
    n = len(mat[0])

    def check(c):
        if c < 10:
            return False
        for i in range(2, int(c ** 0.5) + 1):
            if c % i == 0:
                return False
        return True

    def get_one(x0, y0, x, y):
        res = []
        cur = 0
        while -1 < x0 < m and -1 < y0 < n:
            cur = cur * 10 + mat[x0][y0]
            if check(cur):
                res.append(cur)
            x0 += x
            y0 += y
        return res

    f = defaultdict(int)
    for i in range(m):
        for j in range(n):
            if cur := get_one(i, j, 1, 0):
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, -1, 0)) != -1:
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, 1, 1)) != -1:
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, -1, 1)) != -1:
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, 0, 1)) != -1:
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, 1, -1)) != -1:
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, -1, -1)) != -1:
                for x in cur:
                    f[x] += 1
            if (cur := get_one(i, j, 0, -1)) != -1:
                for x in cur:
                    f[x] += 1

    max_cnt = -1
    max_num = -1
    for num in f:
        if f[num] == max_cnt:
            max_num = max(num, max_num)
        elif f[num] > max_cnt:
            max_cnt = f[num]
            max_num = num
    return max_num


def weekly_contest_386_solution_1(nums: List[int]) -> bool:
    cnt = Counter[int](nums)
    if max(cnt.values()) >= 3:
        return False
    else:
        return True


def biweekly_contest_125_solution_1(nums: List[int], k: int) -> int:
    cnt = 0
    for num in nums:
        if num < k:
            cnt += 1
    return cnt


def biweekly_contest_125_solution_2(nums: List[int], k: int) -> int:
    heapq.heapify(nums)
    cnt = 0
    while nums[0] < k and len(nums) >= 2:
        t1 = heapq.heappop(nums)
        t2 = heapq.heappop(nums)
        t3 = min(t1, t2) * 2 + max(t1, t2)  # t3 = t1*2 + t2
        heapq.heappush(nums, t3)
        cnt += 1
    return cnt


def biweekly_contest_125_solution_3(edges: List[List[int]], signalSpeed: int) -> List[int]:
    n = len(edges) + 1
    f = defaultdict(dict)
    for a, b, w in edges:
        f[a][b] = w
        f[b][a] = w

    @cache
    def dfs(x, p, distance):
        res = 0
        if distance % signalSpeed == 0:
            res += 1
            distance = 0
        # 无效遍历太多
        # for i in range(n):
        #     if i != p and i in f[x]:
        #         res += dfs(i, x, distance + (f[x][i] % signalSpeed))
        for i in f[x]:
            if i != p:
                res += dfs(i, x, distance + (f[x][i] % signalSpeed))
        return res

    count = [0] * n
    for i in range(n):
        total_len = 0
        for j in f[i]:
            res = dfs(j, i, f[i][j])
            count[i] += res * total_len
            total_len += res
    return count


def weekly_contest_387_solution_1(nums: List[int]) -> List[int]:
    arr1 = [nums[0]]
    arr2 = [nums[1]]
    for num in nums[2:]:
        if arr1[-1] > arr2[-1]:
            arr1.append(num)
        else:
            arr2.append(num)
    return arr1 + arr2


def weekly_contest_387_solution_2(grid: List[List[int]], k: int) -> int:
    m = len(grid)
    n = len(grid[0])
    ans = [[0] * (n + 1) for _ in range(m + 1)]
    cnt = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if i == j == 0:
                continue
            ans[i][j] = ans[i - 1][j] + ans[i][j - 1] - ans[i - 1][j - 1] + grid[i - 1][j - 1]
            if ans[i][j] <= k:
                cnt += 1
    return cnt


def weekly_contest_387_solution_3(grid: List[List[int]]) -> int:
    n = len(grid)
    mid = (n // 2) + 1
    cnt = defaultdict(int)
    for i in range(mid):
        cnt[grid[i][i]] += 1
        cnt[grid[i][n - 1 - i]] += 1
        cnt[grid[mid - 1 + i][mid - 1]] += 1
    cnt[grid[mid - 1][mid - 1]] -= 2
    cnt_total = defaultdict(int)
    for i in range(n):
        for j in range(n):
            cnt_total[grid[i][j]] += 1
    cnt_total[0] -= cnt[0]
    cnt_total[1] -= cnt[1]
    cnt_total[2] -= cnt[2]
    tiles = (mid - 1) * 3 + 1
    tiles_total = n * n - tiles

    # i = 0
    changes_1 = tiles - cnt[0]
    changes_2 = min(tiles_total - cnt_total[1], tiles_total - cnt_total[2])
    changes = changes_1 + changes_2
    # i = 1
    changes_1 = tiles - cnt[1]
    changes_2 = min(tiles_total - cnt_total[0], tiles_total - cnt_total[2])
    changes = min(changes, changes_1 + changes_2)
    # i = 2
    changes_1 = tiles - cnt[2]
    changes_2 = min(tiles_total - cnt_total[0], tiles_total - cnt_total[1])
    changes = min(changes, changes_1 + changes_2)

    # for x in cnt_total:
    #     cnt_total[x] -= cnt[x]
    # not_changes = 0
    # for i in range(3):
    #     for j in range(3):
    #         if i != j:
    #             not_changes = max(not_changes,cnt[i]+cnt_total[j])
    # changes = n*n - not_changes
    return changes
