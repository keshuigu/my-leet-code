from testcase import *

if __name__ == '__main__':
    function_name = 'testcase_' + input("Enter problem index:\n")
    getattr(testcase, function_name)()
    # print(test_any(index=2, args=[10]))
