import heapq
from typing import *

if __name__ == '__main__':
    f = DefaultDict[int,list](list)
    f[12].append(3)
    print(f[12])