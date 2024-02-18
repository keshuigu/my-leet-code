import heapq
from typing import *

if __name__ == '__main__':
    f = DefaultDict[int,list](list)
    f[12].append(3)
    print(f[12])
    s = [1,2,3]
    s.sort(reverse=True)
    print(s)