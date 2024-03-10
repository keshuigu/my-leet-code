import bisect
from typing import *

if __name__ == '__main__':
    a = [1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
    b = [1, 1, 2, 3, 2, 3, 3, 4, 4, 5, 8, 6, 6, 7, 7]

    cnt1 = Counter[int](a)
    cnt2 = Counter[int](b)
    print(cnt1 & cnt2)
