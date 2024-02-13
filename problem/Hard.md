- [x] [233.数字1的个数](https://leetcode.cn/problems/number-of-digit-one/)
- [x] [410.分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/)

    - 最大化最小值/最小化最大值 -> 二分查找
    - [二分答案](https://leetcode.cn/problems/split-array-largest-sum/solutions/2613046/er-fen-da-an-fu-ti-dan-pythonjavacgojsru-n5la/)

- [x] [466.统计重复个数](https://leetcode.cn/problems/count-the-repetitions/)

    - [官方题解](https://leetcode.cn/problems/count-the-repetitions/solutions/208874/tong-ji-zhong-fu-ge-shu-by-leetcode-solution/)
    - 鸽笼原理:
      每过一个s1，对应匹配到的s2的index只有|s2|种可能：0-|s2-1|，所以经过|s2|+1个s1，这个s1结束时匹配到的index必然和前面某个s1结束时匹配到的index相同。进一步，只要index“相同”就能找到循环节。

- [x] [514.不含连续1的非负整数](https://leetcode.cn/problems/freedom-trail/)

    - [动态规划入门](https://b23.tv/72onpYq)

- [x] [600.不含连续1的非负整数](https://leetcode.cn/problems/non-negative-integers-without-consecutive-ones/)
- [x] [902.最大为N的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)
- [x] [987.二叉树的垂序遍历](https://leetcode.cn/problems/vertical-order-traversal-of-a-binary-tree)
- [x] [1483.树节点的第K个祖先](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/): LCA
- [x] [1944.队列中可以看到的人数](https://leetcode.cn/problems/number-of-visible-people-in-a-queue/)
- [x] [2376.统计特殊整数](https://leetcode.cn/problems/count-special-integers/)

    - [数位DP模板](https://leetcode.cn/problems/count-special-integers/solutions/1746956/shu-wei-dp-mo-ban-by-endlesscheng-xtgx/)

- [x] [2719.统计整数数目](https://leetcode.cn/problems/count-of-integers/)

    - [数位DP](https://leetcode.cn/problems/count-of-integers/solutions/2601111/tong-ji-zheng-shu-shu-mu-by-leetcode-sol-qxqd/)

- [x] [2809.使数组和小于等于x的最少时间](https://leetcode.cn/problems/minimum-time-to-make-array-sum-at-most-x/)
- [x] [2846.边权重均等查询](https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/)

    - LCA
    - [0x3f的题解](https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/solutions/2424060/lca-mo-ban-by-endlesscheng-j54b/):
      其中cnt为三维数组，`cnt[x][i][j]`代表的是x到2^i祖先的所有边中，权重为j的边的个数

- [x] [3017.按距离统计房屋对数目II](https://leetcode.cn/problems/count-the-number-of-houses-at-a-certain-distance-ii/)
- [x] [3022.给定操作次数内使剩余元素的或值最小](https://leetcode.cn/problems/minimize-or-of-remaining-elements-using-operations/)

  - [拆位, 试填, heq连续子数组合并 from 0x3f](https://leetcode.cn/problems/minimize-or-of-remaining-elements-using-operations/solutions/2622658/shi-tian-fa-pythonjavacgo-by-endlesschen-ysom/)

- [x] [3027.人员站位的方案数 II](https://leetcode.cn/problems/find-the-number-of-ways-to-place-people-ii/)
- [x] [3031.将单词恢复初始状态所需的最短时间 II](https://leetcode.cn/problems/minimum-time-to-revert-word-to-initial-state-ii/)
