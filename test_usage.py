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
    prices = [[1,4,2],[2,2,7],[2,1,3]]
    pr = {(h, w): p for h, w, p in prices}
    print(pr)