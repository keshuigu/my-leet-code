import heapq
from heapq import *
from typing import *


def solution_iq_17_06(n: int) -> int:
    s = str(n)
    length = len(s)
    dp = [[-1] * length for _ in range(length)]

    def f(i, j, limit):
        if i == length:
            return j
        if not limit and dp[i][j] != -1:
            return dp[i][j]
        res = 0
        up = int(s[i]) if limit else 9
        for d in range(up + 1):
            res += f(i + 1, j + (d == 2), limit and d == up)
        if not limit:
            dp[i][j] = res
        return res

    return f(0, 0, True)


def solution_lcp_24(nums: List[int]) -> List[int]:
    MOD = 10 ** 9 + 7
    ans = [0] * len(nums)
    # python默认小根堆，因此left中值取反
    left = []
    right = []
    left_sum = right_sum = 0
    for i, b in enumerate(nums):
        b -= i
        if i % 2 == 0:  # 包含i的前缀长度为奇数 [0,...,i]
            if left:
                # 如果b比left的最大值大，那么插入到right中去
                # 否则插入到left中
                # 此处left_sum值已经删除了left的堆顶，并加入b
                left_sum -= max(-left[0] - b, 0)
            # 无论插入那边，都可以用这个函数
            t = -heappushpop(left, -b)
            heappush(right, t)
            right_sum += t
            # 现在
            ans[i] = (right_sum - right[0] - left_sum) % MOD
        else:  # 前缀长度为偶数,left必定比right少1,往left里插入
            right_sum += max(b - right[0], 0)
            t = heappushpop(right, b)
            left_sum += t
            heappush(left, -t)
            ans[i] = (right_sum - left_sum) % MOD
    return ans


def solution_lcp_30(nums: List[int]) -> int:
    heap = []
    cnt = 0
    cur = 1
    if sum(nums) < 0:
        return -1
    for num in nums:
        if num < 0:
            heapq.heappush(heap, num)
        cur += num
        while heap and cur <= 0:
            tmp = heapq.heappop(heap)
            cur -= tmp
            nums.append(tmp)
            cnt += 1
        # 如果cur<=0,当前num一定为负，heap一定不为空
        # 弹出最大的一定能使得cur为正
        # 不需要加到最后
        # 一定有结果
        # if cur <= 0:
        #     cur -= heapq.heappop(heap)
        #     cnt += 1
    return cnt
