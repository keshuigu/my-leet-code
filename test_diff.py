import re

from testcase import *
import re
import sys


class ListPrinter:
    def __init__(self):
        self.l = []

    def write(self, text):
        self.l.append(text)

    def get_text(self):
        res = ''
        for ans in self.l:
            if ans == '\n':
                continue
            res += '$' + ans
        return res

    def __str__(self):
        return self.l.__str__()


def green_msg(msg):
    return "\033[32m" + msg + "\033[0m"


def red_msg(msg):
    return "\033[1;31m" + msg + "\033[0m"


if __name__ == '__main__':
    function_name = 'testcase_' + input("Enter problem index:\n")
    pattern = re.compile(f"^{function_name}(_[0-9]+)?$")
    diffs = []
    ref = ''
    for name in dir(testcase):
        obj = re.match(pattern, name)
        if obj:
            tmp = ListPrinter()
            sys.stdout = tmp
            function_name = obj.group()
            getattr(testcase, function_name)()
            sys.stdout = sys.__stdout__
            if function_name[-2:] == '_0':
                ref = tmp.get_text()
            else:
                diffs.append(tmp.get_text())
    print(green_msg("----ref answer----"))
    print(green_msg(ref))
    print(green_msg("------------------"))
    rl = len(ref)
    for answer in diffs:
        al = len(answer)
        cl = min(al, rl)
        p_msg = []
        for i in range(cl):
            if answer[i] != ref[i]:
                p_msg.append(red_msg(answer[i]))
            else:
                p_msg.append(green_msg(answer[i]))
        if cl < al:
            p_msg.append(red_msg(answer[cl + 1:al]))
        print(''.join(p_msg))
