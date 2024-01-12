from typing import *
from .data_struct import *
from .method import *


def solution_2235(num1: int, num2: int) -> int:
    return num1 + num2


def solution_2487(head: Optional[ListNode]) -> Optional[ListNode]:
    # 双向链表
    if head is None or head.next is None:
        return head
    d_head = DListNode(head)
    p = head
    q = d_head
    while p.next is not None:
        q.next = DListNode(p.next)
        q.next.prev = q
        q = q.next
        p = p.next
    head = q.val
    q = q.prev
    while q is not None:
        if q.val.val >= head.val:
            temp = head
            head = q.val
            head.next = temp
        q = q.prev
    return head


def solution_2487_2(head: Optional[ListNode]) -> Optional[ListNode]:
    # 反转链表
    def reverse(head_re: ListNode) -> ListNode:
        dummy = ListNode(head_re.val)
        while head_re is not None:
            p_re = head_re
            head_re = head_re.next
            p_re.next = dummy.next
            dummy.next = p_re
        return dummy.next

    head = reverse(head)
    # 找左侧有更大数值的节点删除
    p = head
    while p is not None and p.next is not None:
        if p.next.val < p.val:
            p.next = p.next.next
        else:
            p = p.next
    return reverse(head)


def solution_2487_3(head: Optional[ListNode]) -> Optional[ListNode]:
    # 单调栈
    stack = []
    p = head
    rest = ListNode(-1)
    while p is not None:
        while not len(stack) == 0:
            if p.val > stack[len(stack) - 1].val:
                stack.pop()
            else:
                stack[len(stack) - 1].next = p
                stack.append(p)
                break
        if len(stack) == 0:
            stack.append(p)
            rest.next = p
        p = p.next
    return rest.next


def solution_2397(matrix: List[List[int]], numSelect: int) -> int:
    com_arrays = com_iteration(len(matrix[0]), numSelect)
    count_arrays = []
    for com_item in com_arrays:
        matrix_copy = []
        for i in range(len(matrix)):
            matrix_copy.append(list(matrix[i]))
        for i in range(len(matrix_copy)):
            for j in range(len(matrix_copy[0])):
                if j + 1 in com_item:
                    matrix_copy[i][j] = 0
        count = 0
        for row in matrix_copy:
            if sum(row) == 0:
                count += 1
        count_arrays.append(count)
    return max(count_arrays)


def solution_2397_2(matrix: List[List[int]], numSelect: int) -> int:
    # 预处理矩阵,压缩成m*1的数组,用二进制数表示
    m = len(matrix)
    n = len(matrix[0])
    mask = [0] * m
    for i in range(m):
        for j in range(n):
            mask[i] += matrix[i][j] << (n - j - 1)
    # 用一个整数来表述所有组合,不满足的跳过即可
    limit = 1 << n
    res = 0
    for i in range(limit):
        if pop_count(i) != numSelect:
            continue
        count = 0
        for j in range(m):
            if mask[j] & i == mask[j]:
                count += 1
        res = max(res, count)
    return res


def solution_2397_3(matrix: List[List[int]], numSelect: int) -> int:
    # 预处理矩阵,压缩成m*1的数组,用二进制数表示
    m = len(matrix)
    n = len(matrix[0])
    mask = [0] * m
    for i in range(m):
        for j in range(n):
            mask[i] += matrix[i][j] << (n - j - 1)
    # Gosper's Hack
    res = 0
    limit = 1 << n
    cur = (1 << numSelect) - 1
    while cur < limit:
        count = 0
        for j in range(m):
            if mask[j] & i == mask[j]:
                count += 1
        res = max(res, count)
        lb = cur & -cur  # 取最低位的1
        r = cur + lb  # 在cur最低位的1上加1
        cur = ((r ^ cur) >> count_trailing_zeros(lb) + 2) | r
    return res


def solution_2085(words1: List[str], words2: List[str]) -> int:
    f = {}
    for word in words1:
        if word not in f:
            f[word] = 1
        else:
            f[word] += 1
    for word in f:
        if f[word] > 1:
            f[word] = 3
    for word in words2:
        if word in f:
            f[word] += 1
    count = 0
    for word in f:
        if f[word] == 2:
            count += 1
    return count
