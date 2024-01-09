import os

from testcase import *
import re
import time
import sys

if __name__ == '__main__':
    function_name = 'testcase_' + input("Enter problem index:\n")
    pattern = re.compile(f"^{function_name}(_[0-9]+)?$")
    for name in dir(testcase):
        obj = re.match(pattern, name)
        if obj:
            sys.stdout = open(os.devnull, 'w')
            function_name = obj.group()
            start_time = time.time()
            for _ in range(100000):
                getattr(testcase, function_name)()
            end_time = time.time()
            execution_time = end_time - start_time
            sys.stdout = sys.__stdout__
            print(f'{function_name}: {execution_time}s')
    # print(test_any(index=2, args=[10]))
