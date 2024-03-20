import heapq

from solution import gospers_hack


def solution_367(num: int) -> bool:
    x0 = num
    x1 = x0 - (x0 * x0 - num) / (2 * x0)
    while x0 - x1 > 10e-6 or x1 - x0 > 10e-6:
        x0 = x1
        x1 = x0 - (x0 * x0 - num) / (2 * x0)
    return x1


def solution_367_2(num: int) -> bool:
    x0 = 0.1
    x1 = x0 - (x0 * x0 - num) / (2 * x0)
    while x0 - x1 > 10e-6 or x1 - x0 > 10e-6:
        x0 = x1
        x1 = x0 - (x0 * x0 - num) / (2 * x0)
    return x1


if __name__ == '__main__':
    print(gospers_hack(0, 6))