import random
import time

if __name__ == '__main__':
    now = time.time()
    for i in range(1000000):
        a = random.randint(1, 100)
        b = a if a < 50 else 50
    print(f'time: {time.time() - now:.2f}s')
    now = time.time()
    for i in range(1000000):
        a = random.randint(1, 100)
        b = min(a, 50)
    print(f'time: {time.time() - now:.2f}s')
