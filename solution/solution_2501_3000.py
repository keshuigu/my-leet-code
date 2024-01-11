from typing import *
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
