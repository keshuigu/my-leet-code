import random
import time
from solution.data_struct import *
from solution.method import *

if __name__ == '__main__':
    s = "A man, a plan, a canal: Panama"
    tmp = ""
    for i in range(len(s)):
        num = ord(s[i])
        if 65 <= num <= 90:
            tmp += chr(num + 32)
        if 97 <= num <= 122:
            tmp += s[i]
    print(tmp)
