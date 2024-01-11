from testcase import *
import re
import sys
import os

if __name__ == '__main__':
    os.system("egrep '(^[0-9]{3}-[0-9]{3}-[0-9]{4}$)|(^\([0-9]{3}\) [0-9]{3}-[0-9]{4}$)' resources/file.txt")
    os.system("sed -n -r  '/^([0-9]{3}-|\([0-9]{3}\) )[0-9]{3}-[0-9]{4}$/p' resources/file.txt")
