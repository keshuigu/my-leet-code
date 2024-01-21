from typing import *


def solution_100191(word: str) -> int:
    if len(word) <= 8:
        return len(word)
    if len(word) <= 16:
        return (len(word) - 8) * 2 + 8
    if len(word) <= 24:
        return (len(word) - 16) * 3 + 24
    return (len(word) - 24) * 4 + 48


def solution_100188(n: int, x: int, y: int) -> List[int]:
    f = {}
    res = [0] * n
    if x != y:
        f[(x-1, y-1)] = 1
        f[(y-1, x-1)] = 1
        res[0] += 2
    for i in range(n - 1):
        f[(i, i + 1)] = 1
        f[(i + 1, i)] = 1
        res[0] += 2
    for j in range(2, n):
        new_f = {}
        for (x1, y1) in f:
            for (x2, y2) in f:
                if (x1, y2) in new_f:
                    continue
                if y1 == x2 and x1 != y2 and (x1, y2) not in f and f[(x1, y1)] + f[x2, y2] == j:
                    new_f[(x1, y2)] = j
                    res[j - 1] += 1
        for k in new_f:
            f[k] = j
    return res


def solution_100192(word: str) -> int:
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
