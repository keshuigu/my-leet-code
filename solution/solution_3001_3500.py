from itertools import accumulate
from typing import *
from .data_struct import *
from .method import *
from .competition_solution import *


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
