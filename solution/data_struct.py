import queue


class Stack:
    def __init__(self, max_size):
        self.max = max_size
        self.items = []
        self.top = 0  # 栈顶指针指向空

    def is_empty(self):
        return self.top == 0

    def is_full(self):
        return self.top == self.max

    def push(self, item):
        if self.is_full():
            raise Exception('Stack Overflow')
        else:
            self.items.append(item)
            self.top += 1

    def pop(self):
        if self.is_empty():
            raise Exception('Stack Underflow')
        else:
            self.top -= 1
            return self.items.pop()

    def peek(self):
        if self.is_empty():
            raise Exception('Stack Underflow')
        else:
            return self.items[self.top - 1]

    def size(self):
        return len(self.items)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __str__(self):
        ret = str(self.val)
        temp = self.next
        while temp is not None:
            ret += '->' + str(temp.val)
            temp = temp.next
        return ret


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
