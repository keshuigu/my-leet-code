1. `二叉搜索树`:它或者是一棵空树，或者是具有下列性质的二叉树： 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
   若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值； 它的左、右子树也分别为二叉排序树。
2. `高度平衡`:二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。
3. [`欧几里得算法`](https://baike.baidu.com/item/%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97%E7%AE%97%E6%B3%95/1647675?fromtitle=%E8%BE%97%E8%BD%AC%E7%9B%B8%E9%99%A4%E6%B3%95&fromid=4625352&fr=aladdin):
   又称辗转相除法，是指用于计算两个非负整数a，b的最大公约数。
4. [`前缀树`](https://zhuanlan.zhihu.com/p/420663173):Trie树或者说前缀树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键
5. `优先队列`：
6. `前缀和`：
7. [`floyd`](https://zhuanlan.zhihu.com/p/339542626)
8. [`差分数组`](https://leetcode.cn/problems/car-pooling/solutions/2550264/suan-fa-xiao-ke-tang-chai-fen-shu-zu-fu-9d4ra/)

    1. 对于数组a,定义差分数组为d[0]=a[0],d[i]=a[i]-a[i-1]
    2. 性质1: 从左到右累加d中的元素,可以得到数组a
    3. 性质2: 下面两个操作等价:
        - 把d[i]增加x,把d[j+1]减少x
        - 把a的子数组a[i]到a[j]都加上x

9. `LCA`:[参考资料1](https://oi-wiki.org/graph/lca/), [参考题解](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/solutions/2305895/mo-ban-jiang-jie-shu-shang-bei-zeng-suan-v3rw/)
10. [`排序不等式`](https://zh.wikipedia.org/wiki/%E6%8E%92%E5%BA%8F%E4%B8%8D%E7%AD%89%E5%BC%8F)
11. [离散化+二维前缀和](https://www.cnblogs.com/tyriis/p/15362478.html)
12. [二维差分](https://leetcode.cn/problems/stamping-the-grid/solutions/1199642/wu-nao-zuo-fa-er-wei-qian-zhui-he-er-wei-zwiu/)
13. [heapify的时间复杂度为什么是O(n)](https://www.jianshu.com/p/147bb9ee41b1):同一层的节点最多交换的次数都是相同的,第1层最多交换k-1次,第2层最多交换k-2次...