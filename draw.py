from itertools import accumulate

import matplotlib.pyplot as plt
import pandas as pd



if __name__ == '__main__':
    data = pd.read_csv('resources/scores.csv')
    x=data['name']
    origin_y=data['changes']
    y=list(accumulate(origin_y,initial=0))
    plt.plot(x,y[1:])
    plt.xlabel('contest')
    plt.ylabel('Rank scores')
    plt.title('keshuigu\'s path')
    plt.show()