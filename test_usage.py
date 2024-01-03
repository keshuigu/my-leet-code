import random
import time
from solution.data_struct import *

if __name__ == '__main__':
    head = ListNode(0, ListNode(1, ListNode(2, ListNode(3, ListNode(4)))))
    d_head = DListNode(head)
    p = head
    q1 = d_head
    while p.next is not None:
        q1.next = DListNode(p.next)
        q1.next.prev = q1
        q1 = q1.next
        p = p.next
    print(d_head)
