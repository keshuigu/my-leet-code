from solution import *


def test_any(**kwargs):
    func_name = "solution_" + str(kwargs.get("index"))
    args = kwargs.get("args")
    return getattr(solution, func_name)(*args)
