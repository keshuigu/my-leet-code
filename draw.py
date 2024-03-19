from itertools import accumulate

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('resources/scores.csv')
    x_label = data['name']
    origin_y = data['changes']
    y = list(accumulate(origin_y, initial=0))[1:]
    x = range(len(x_label))
    plt.plot(x, y)
    plt.scatter(x, y)
    for i in x:
        # if i == 0:
        #     changes = y[i]
        # else:
        #     changes = y[i] - y[i - 1]
        # changes = "+" + str(changes) if changes > 0 else str(changes)
        # plt.text(x[i], y[i], f"{y[i]}({changes})")
        plt.text(x[i], y[i], f"{y[i]}")
    plt.xticks(x, x_label.values, rotation=45)
    plt.xlabel('contest')
    plt.ylabel('Rank scores')
    plt.title('keshuigu\'s path')
    plt.savefig('./asserts/path.png')
