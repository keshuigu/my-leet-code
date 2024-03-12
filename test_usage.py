import bisect
from typing import *

if __name__ == '__main__':
    n = 4
    m = 2 ** 31 - 1
    while n < m:
        print(n)
        n *= 4
