from testcase import *
import re
import sys
import os

if __name__ == '__main__':
    sys.stdout = open(os.devnull, 'w')
    for _ in range(100000):
        testcase_466_2()
    sys.stdout = sys.__stdout__
