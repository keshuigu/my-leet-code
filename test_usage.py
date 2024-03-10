import bisect
from typing import *

if __name__ == '__main__':
    s = ['ca','ab','abc','ba']
    s.sort(key=lambda x:(len(x),x))
    print(s)
