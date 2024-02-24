import bisect
from typing import *

if __name__ == '__main__':
    f = DefaultDict[int, list](list)
    f[12].append(3)
    print(f[12])
    s = [1, 2, 3]
    s.sort(reverse=True)
    s = [1, 2, 3, 4, 5, 7]
    print(s)
    x = bisect.bisect_left(s, 6)
    print(x)
    print(s)
