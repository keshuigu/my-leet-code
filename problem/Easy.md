# Easy

- [x] [1.两数之和](https://leetcode.cn/problems/two-sum/)
- [x] [9.回文数](https://leetcode.cn/problems/palindrome-number/)
- [x] [13.罗马数字转整数](https://leetcode.cn/problems/roman-to-integer/)
- [x] [14.最长公共前缀](https://leetcode.cn/problems/longest-common-prefix/)
- [x] [20.有效的括号](https://leetcode.cn/problems/valid-parentheses/)
- [x] [21.合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)
- [x] [26.删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)
- [x] [27.移除元素](https://leetcode.cn/problems/remove-element/)
- [x] [28.找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)

    - 考虑使用KMP算法：[参考1](https://zhuanlan.zhihu.com/p/83334559), 算法导论32.4节

- [x] [35.搜索插入位置](https://leetcode.cn/problems/search-insert-position/)
- [x] [35.最后一个单词的长度](https://leetcode.cn/problems/length-of-last-word/)
- [x] [66.加一](https://leetcode.cn/problems/search-insert-position/)
- [x] [67.二进制求和](https://leetcode.cn/problems/add-binary/)
- [x] [69.x的平方跟](https://leetcode.cn/problems/sqrtx/)
- [x] [70.爬楼梯](https://leetcode.cn/problems/climbing-stairs)

    - [参考题解](https://leetcode.cn/problems/climbing-stairs/solutions/286022/pa-lou-ti-by-leetcode-solution/)
    - [标量快速幂,矩阵快速幂](https://zhuanlan.zhihu.com/p/95902286)
    - ```python
      def quick_pow(a:int,n:int)->int:
          ans = 1 
          while n > 0: # 指数不为0
              if n & 1: # 二进制当前为1,说明结果需要乘以当前底数的2的若干次方
                  ans *= a
              a *= a  # 计算下一个底数的2的若干次方
              n = n >> 1
          return ans
      ```
    - 实际上对于任意类型的可以进行乘法和满足结合律的a,只需要重载a的乘法运算即可

- [x] [83.删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)
- [x] [88.合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)
- [x] [94.二叉树的中序遍历](https://leetcode.cn/problems/binary-tree-inorder-traversal/)

    - [Morris](https://leetcode.cn/problems/binary-tree-inorder-traversal/solutions/412886/er-cha-shu-de-zhong-xu-bian-li-by-leetcode-solutio/)

- [x] [100.相同的树](https://leetcode.cn/problems/same-tree/)
- [x] [101.对称二叉树](https://leetcode.cn/problems/symmetric-tree/)
- [x] [104.二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)
- [x] [108.将有序数组转换为二叉搜索树](https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/)
- [x] [110.平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)
- [x] [111.二叉树的最小深度](https://leetcode.cn/problems/minimum-depth-of-binary-tree/)
- [x] [118.杨辉三角](https://leetcode.cn/problems/pascals-triangle/)
- [x] [119.杨辉三角Ⅱ](https://leetcode.cn/problems/pascals-triangle-ii/)
- [x] [121.买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)
- [x] [125.验证回文串](https://leetcode.cn/problems/valid-palindrome/)
- [x] [136.只出现一次的数组](https://leetcode.cn/problems/single-number/)

    - 异或运算

- [x] [141.环形链表](https://leetcode.cn/problems/linked-list-cycle/)
- [x] [144.二叉树的前序遍历](https://leetcode.cn/problems/binary-tree-preorder-traversal/)
- [x] [145.二叉树的后序遍历](https://leetcode.cn/problems/binary-tree-postorder-traversal/)
- [x] [160.相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)
- [x] [168.Excel表列名称](https://leetcode.cn/problems/excel-sheet-column-title/)

    - 没有0的进制转换,映射时多多注意
    - 对于此题,每位-1即成为余数,对于不同的题目,考虑怎么设置映射使得转换为余数的计算

- [x] [169.多数元素](https://leetcode.cn/problems/majority-element/)

    - 摩尔投票算法
    - 题解证明有误,count和value并非一一对应,但是最后count必定大于0,且对应candidate
    - ```
    # 题解给出的证明
    nums:      [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 5, 7, 7 | 7, 7, 7, 7]
    candidate:  7  7  7  7  7  7   5  5   5  5  5  5   7  7  7  7
    count:      1  2  1  2  1  0   1  0   1  2  1  0   1  2  3  4
    value:      1  2  1  2  1  0  -1  0  -1 -2 -1  0   1  2  3  4
    # 1个5换成1
    nums:      [7, 7, 5, 7, 5, 1 | 5, 7 | 5, 1, 7, 7 | 7, 7, 7, 7]
    candidate:  7  7  7  7  7  7   5  5   5  5  7  7   7  7  7  7
    count:      1  2  1  2  1  0   1  0   1  0  1  2   3  4  5  6
    value:      1  2  1  2  1  0  -1  0  -1 -2 -1  0   1  2  3  4
    ```

- [x] [171.Excel表列序号](https://leetcode.cn/problems/excel-sheet-column-number/)

    - 从无0转有0的进制,简单很多

- [x] [175.组合两个表](https://leetcode.cn/problems/combine-two-tables/)
- [x] [181.超过经理收入的员工](https://leetcode.cn/problems/employees-earning-more-than-their-managers/)
- [x] [182.查找重复的电子邮箱](https://leetcode.cn/problems/duplicate-emails/)
- [x] [183.从不订购的客户](https://leetcode.cn/problems/customers-who-never-order/)
- [x] [190.颠倒二进制位](https://leetcode.cn/problems/reverse-bits/)
- [x] [191.位1的个数](https://leetcode.cn/problems/number-of-1-bits/)
- [x] [193.有效电话号码](https://leetcode.cn/problems/valid-phone-numbers/)
- [x] [195.第十行](https://leetcode.cn/problems/tenth-line/)
- [x] [196.删除重复的电子邮箱](https://leetcode.cn/problems/delete-duplicate-emails/)
- [x] [197.上升的温度](https://leetcode.cn/problems/rising-temperature/)
- [x] [202.快乐数](https://leetcode.cn/problems/happy-number/)
- [x] [203.移除链表元素](https://leetcode.cn/problems/remove-linked-list-elements/)
- [x] [205.同构字符串](https://leetcode.cn/problems/isomorphic-strings/)
- [x] [206.反转链表](https://leetcode.cn/problems/reverse-linked-list/)
- [x] [217.存在重复元素](https://leetcode.cn/problems/contains-duplicate/)
- [x] [219.存在重复元素Ⅱ](https://leetcode.cn/problems/contains-duplicate-ii/)
- [x] [222.完全二叉树的节点个数](https://leetcode.cn/problems/count-complete-tree-nodes/)

    - [二分查找](https://leetcode.cn/problems/count-complete-tree-nodes/solutions/495655/wan-quan-er-cha-shu-de-jie-dian-ge-shu-by-leetco-2/)

- [x] [225.用队列实现栈](https://leetcode.cn/problems/implement-stack-using-queues/)
- [x] [226.翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)
- [x] [228.汇总区间](https://leetcode.cn/problems/summary-ranges/)
- [x] [231.2的幂](https://leetcode.cn/problems/power-of-two/)
- [x] [232.用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)
- [x] [234.回文链表](https://leetcode.cn/problems/palindrome-linked-list/)
- [x] [240.用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)
- [x] [242.有效的字母异位词](https://leetcode.cn/problems/valid-anagram/)
- [x] [257.二叉树的所有路径](https://leetcode.cn/problems/binary-tree-paths/)
- [x] [258.各位相加](https://leetcode.cn/problems/valid-anagram/)

    - [数根](https://en.wikipedia.org/wiki/Digital_root)

- [x] [263.丑数](https://leetcode.cn/problems/ugly-number/)
- [x] [268.丢失的数字](https://leetcode.cn/problems/missing-number/)
- [x] [278.第一个错误的版本](https://leetcode.cn/problems/first-bad-version/)
- [x] [283.移动零](https://leetcode.cn/problems/move-zeroes/)
- [x] [290.单词规律](https://leetcode.cn/problems/word-pattern/)
- [x] [292.Nim游戏](https://leetcode.cn/problems/nim-game/)

    - [Nim游戏](https://baike.baidu.com/item/Nim%E6%B8%B8%E6%88%8F/6737105)

- [x] [303.区域和检索-数组不可变](https://leetcode.cn/problems/range-sum-query-immutable/)
- [x] [326.3的幂](https://leetcode.cn/problems/power-of-three/)
- [x] [338.比特位计数](https://leetcode.cn/problems/counting-bits/)
- [x] [342.比特位计数](https://leetcode.cn/problems/power-of-four/)
- [x] [344.反转字符串](https://leetcode.cn/problems/reverse-string/)
- [x] [345.反转字符串中的元音字母](https://leetcode.cn/problems/reverse-vowels-of-a-string/)
- [x] [349.两个数组的交集](https://leetcode.cn/problems/intersection-of-two-arrays/)
- [x] [350.两个数组的交集 II](https://leetcode.cn/problems/intersection-of-two-arrays-ii/)
- [x] [367.有效的完全平方数](https://leetcode.cn/problems/valid-perfect-square/)
- [x] [374.猜数字大小](https://leetcode.cn/problems/guess-number-higher-or-lower/)
- [x] [383.赎金信](https://leetcode.cn/problems/ransom-note/)
- [x] [383.字符串中的第一个唯一字符](https://leetcode.cn/problems/first-unique-character-in-a-string/)
- [x] [389.找不同](https://leetcode.cn/problems/find-the-difference/)
- [x] [392.判断子序列](https://leetcode.cn/problems/is-subsequence/)
- [x] [401.二进制手表](https://leetcode.cn/problems/binary-watch/)
- [x] [404.左叶子之和](https://leetcode.cn/problems/fizz-buzz/)
- [x] [405.数字转换为十六进制](https://leetcode.cn/problems/convert-a-number-to-hexadecimal/)
- [x] [412.Fizz Buzz](https://leetcode.cn/problems/fizz-buzz/)
- [x] [589.N叉树的前序遍历](https://leetcode.cn/problems/n-ary-tree-preorder-traversal/)
- [x] [590.N叉树的后序遍历](https://leetcode.cn/problems/n-ary-tree-preorder-traversal/)
- [x] [704.二分查找](https://leetcode.cn/problems/binary-search/)
- [x] [876.链表的中间节点](https://leetcode.cn/problems/n-ary-tree-postorder-traversal/)
- [x] [983.二叉搜索树的范围和](https://leetcode.cn/problems/range-sum-of-bst/)
- [x] [993.二叉树的堂兄弟节点](https://leetcode.cn/problems/cousins-in-binary-tree/)
- [x] [1154.一年中的第几天](https://leetcode.cn/problems/day-of-the-year/)
- [x] [1185.一周中的第几天](https://leetcode.cn/problems/day-of-the-week/)
- [x] [1342.将数字变成0的操作次数](https://leetcode.cn/problems/number-of-steps-to-reduce-a-number-to-zero/)

    - [计算二进制0的个数和1的个数](https://leetcode.cn/problems/number-of-steps-to-reduce-a-number-to-zero/solutions/1237903/jiang-shu-zi-bian-cheng-0-de-cao-zuo-ci-ucaa4/)
    - [求1的个数的参考](https://zhuanlan.zhihu.com/p/161927442)

- [x] [1379.找出克隆二叉树中的相同节点](https://leetcode.cn/problems/find-a-corresponding-node-of-a-binary-tree-in-a-clone-of-that-tree/)
- [x] [1480.一维数组的动态和](https://leetcode.cn/problems/running-sum-of-1d-array/)
- [x] [1672.最富有客户的资产总量](https://leetcode.cn/problems/richest-customer-wealth/)
- [x] [2085.统计出现过一次的公共字符串](https://leetcode.cn/problems/count-common-words-with-one-occurrence/)
- [x] [2235.两整数相加](https://leetcode.cn/problems/add-two-integers/)
- [x] [2529.正整数和负整数的最大计数](https://leetcode.cn/problems/maximum-count-of-positive-integer-and-negative-integer/)
- [x] [2549.找出不同元素数目差数组](https://leetcode.cn/problems/count-distinct-numbers-on-board/)
- [x] [2670.找出不同元素数目差数组](https://leetcode.cn/problems/find-the-distinct-difference-array/)

    - 前后缀分解

- [x] [2696.删除字串后的字符串最小长度](https://leetcode.cn/problems/minimum-string-length-after-removing-substrings/)
- [x] [2706.购买两块巧克力](https://leetcode.cn/problems/buy-two-chocolates/)
- [x] [2744.最大字符串配对数目](https://leetcode.cn/problems/find-maximum-number-of-string-pairs/)
- [x] [2765.最长交替子数组](https://leetcode.cn/problems/longest-alternating-subarray/)

    - [分组循环](https://leetcode.cn/problems/longest-alternating-subarray/solutions/2615916/jiao-ni-yi-ci-xing-ba-dai-ma-xie-dui-on-r57bz/)

- [x] [2788.按分隔符拆分字符串](https://leetcode.cn/problems/split-strings-by-separator/)
- [x] [2810.故障键盘](https://leetcode.cn/problems/faulty-keyboard/)
- [x] [2859.计算K置位下标对应元素的和](https://leetcode.cn/problems/sum-of-values-at-indices-with-k-set-bits/)
- [x] [2864.最大二进制奇数](https://leetcode.cn/problems/maximum-odd-binary-number/)
- [x] [2908.元素和最小的山形三元组 I](https://leetcode.cn/problems/minimum-sum-of-mountain-triplets-i/)
- [x] [2917.找出数组中的K-or值](https://leetcode.cn/problems/find-the-k-or-of-an-array/)
- [x] [3014.输入单词需要的最少按键次数I](https://leetcode.cn/problems/minimum-number-of-pushes-to-type-word-i/)
- [x] [3019.按键变更的次数](https://leetcode.cn/problems/number-of-changing-keys/)
- [x] [3024.三角形类型](https://leetcode.cn/problems/type-of-triangle-ii/)
- [x] [3028.边界上的蚂蚁](https://leetcode.cn/problems/ant-on-the-boundary/)
- [x] [3033.修改矩阵](https://leetcode.cn/problems/modify-the-matrix/description/)
- [x] [3038.相同分数的最大操作数目 I](https://leetcode.cn/problems/maximum-number-of-operations-with-the-same-score-i/)
- [x] [3042.统计前后缀下标对 I](https://leetcode.cn/problems/count-prefix-and-suffix-pairs-i/)
- [x] [3046.分割数组](https://leetcode.cn/problems/split-the-array/)
- [x] [3065.超过阈值的最少操作数 I](https://leetcode.cn/problems/minimum-operations-to-exceed-threshold-value-i/)
- [x] [3069.将元素分配到两个数组中 I](https://leetcode.cn/problems/distribute-elements-into-two-arrays-i/)
- [x] [3074.重新分装苹果](https://leetcode.cn/problems/apple-redistribution-into-boxes/)
- [x] [3079.求出加密整数的和](https://leetcode.cn/problems/find-the-sum-of-encrypted-integers/)
- [x] [3084.字符串及其反转中是否存在同一子字符串](https://leetcode.cn/problems/existence-of-a-substring-in-a-string-and-its-reverse/)
- [x] [3090.每个字符最多出现两次的最长子字符串](https://leetcode.cn/problems/maximum-length-substring-with-two-occurrences/)
- [x] [3095.或值至少K的最短子数组 I](https://leetcode.cn/problems/shortest-subarray-with-or-at-least-k-i/)
- [x] [3099.哈沙德数](https://leetcode.cn/problems/harshad-number/)
- [x] [3105.最长的严格递增或递减子数组](https://leetcode.cn/problems/longest-strictly-increasing-or-strictly-decreasing-subarray/)
