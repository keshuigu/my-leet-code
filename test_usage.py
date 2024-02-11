import heapq
from typing import Counter

if __name__ == '__main__':
    words = ["aac", "def", "ghi", "jkl"]
    cnt = Counter[int]()
    for word in words:
        cnt += Counter[int](word)
    print(cnt)
