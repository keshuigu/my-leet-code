from typing import *
from .data_struct import *


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
