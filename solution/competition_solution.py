import math
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
    for num in nums:
        cnt = 2
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


def solution_100179(nums: List[int], k: int) -> int:
    # TODO
    ...


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
    # TODO
    ...


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


def solution_100189_2(image: List[List[int]], threshold: int) -> List[List[int]]:
    # TODO
    ...


def solution_100203_2(word: str, k: int) -> int:
    # TODO
    ...


def solution_100203(word: str, k: int) -> int:
    return solution_100204(word, k)
