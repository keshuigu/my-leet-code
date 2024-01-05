- [x] [466.统计重复个数](https://leetcode.cn/problems/count-the-repetitions/)

    - [官方题解](https://leetcode.cn/problems/count-the-repetitions/solutions/208874/tong-ji-zhong-fu-ge-shu-by-leetcode-solution/)
    - 鸽笼原理:每过一个s1，对应匹配到的s2的index只有|s2|种可能：0-|s2-1|，所以经过|s2|+1个s1，这个s1结束时匹配到的index必然和前面某个s1结束时匹配到的index相同。进一步，只要index“相同”就能找到循环节。
- [x] [1944.队列中可以看到的人数](https://leetcode.cn/problems/number-of-visible-people-in-a-queue/)