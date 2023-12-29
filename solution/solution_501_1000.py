from typing import *
from .data_struct import *


def solution_876(head: Optional[ListNode]) -> Optional[ListNode]:
    # 快慢指针
    p = q = head
    while p is not None and p.next is not None:
        p = p.next.next
        q = q.next
    return q
