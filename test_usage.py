import bisect
from typing import *

if __name__ == '__main__':
    title = "capiTalIze tHe titLe"
    words = title.split(" ")
    n = len(words)
    for i in range(n):
        words[i] = words[i].lower()
        if len(words[i]) >= 2:
            words[i] = chr(ord(words[i][0]) - 32) + words[i][1:]
    print(words)
