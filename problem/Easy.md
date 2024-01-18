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
- [x] [290.单词规律](https://leetcode.cn/problems/word-pattern/)
- [x] [383.赎金信](https://leetcode.cn/problems/ransom-note/)
- [x] [412.Fizz Buzz](https://leetcode.cn/problems/fizz-buzz/)
- [x] [876.链表的中间节点](https://leetcode.cn/problems/middle-of-the-linked-list/)
- [x] [1154.一年中的第几天](https://leetcode.cn/problems/day-of-the-year/)
- [x] [1185.一周中的第几天](https://leetcode.cn/problems/day-of-the-week/)
- [x] [1342.将数字变成0的操作次数](https://leetcode.cn/problems/number-of-steps-to-reduce-a-number-to-zero/)

    - [计算二进制0的个数和1的个数](https://leetcode.cn/problems/number-of-steps-to-reduce-a-number-to-zero/solutions/1237903/jiang-shu-zi-bian-cheng-0-de-cao-zuo-ci-ucaa4/)
    - [求1的个数的参考](https://zhuanlan.zhihu.com/p/161927442)

- [x] [1480.一维数组的动态和](https://leetcode.cn/problems/running-sum-of-1d-array/)
- [x] [1672.最富有客户的资产总量](https://leetcode.cn/problems/richest-customer-wealth/)
- [x] [2085.统计出现过一次的公共字符串](https://leetcode.cn/problems/count-common-words-with-one-occurrence/)
- [x] [2235.两整数相加](https://leetcode.cn/problems/add-two-integers/)
- [x] [2696.删除字串后的字符串最小长度](https://leetcode.cn/problems/minimum-string-length-after-removing-substrings/)
- [x] [2706.购买两块巧克力](https://leetcode.cn/problems/buy-two-chocolates/)
- [x] [2744.最大字符串配对数目](https://leetcode.cn/problems/find-maximum-number-of-string-pairs/)
