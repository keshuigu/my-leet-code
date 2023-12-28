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