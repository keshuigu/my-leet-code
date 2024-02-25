import bisect
from typing import *

if __name__ == '__main__':
    a= [[1,2],[0,4]]
    b = [[4,7],[0,8]]
    s = list(zip(a,b))
    s.sort(key=lambda x:x[0][0])
    print(s)

