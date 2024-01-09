import queue
from typing import *


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


class DListNode:
    def __init__(self, val=0, next=None, prev=None):
        self.val = val
        self.next = next
        self.prev = prev

    def __str__(self):
        ret = str(self.val)
        temp = self.next
        tail = None
        while temp is not None:
            ret += ' => ' + str(temp.val)
            tail = temp
            temp = temp.next
        ret += '\n'
        ret += str(tail.val)
        tail = tail.prev
        while tail is not None:
            ret += ' <= ' + str(tail.val)
            tail = tail.prev
        return ret


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        bfs_q = queue.Queue()
        ret = []
        bfs_q.put(self)
        while not bfs_q.empty():
            t = bfs_q.get()
            if t is not None:
                ret.append(t.val)
                bfs_q.put(t.left)
                bfs_q.put(t.right)
        return str(ret)


class MyTrie:
    def __init__(self):
        self.dict = {}
        self.leave = False

    def insert(self, word: str) -> None:
        if len(word) == 0:
            self.leave = True
            return
        tmp = word[0]
        if tmp not in self.dict:
            self.dict[tmp] = Trie()
        self.dict[tmp].insert(word[1:])

    def search(self, word: str) -> bool:
        if len(word) == 0:
            return self.leave
        tmp = word[0]
        if tmp not in self.dict:
            return False
        return self.dict[tmp].search(word[1:])

    def starts_with(self, prefix: str) -> bool:
        if len(prefix) == 0:
            return True
        tmp = prefix[0]
        if tmp not in self.dict:
            return False
        return self.dict[tmp].starts_with(prefix[1:])


def track(node: Optional["Trie"], ch: str) -> (Optional["Trie"], bool):
    tmp = ord(ch) - ord("a")
    if not node or not node.children[tmp]:
        return None, False
    node = node.children[tmp]
    return node, node.leave


class Trie:  # 仅针对26个小写字母
    def __init__(self):
        self.children: List[Union[None, Trie]] = [None] * 26
        self.leave = False

    def search_prefix(self, prefix: str) -> Optional["Trie"]:
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return None
            node = node.children[ch]
        return node

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.leave = True

    def search(self, word: str) -> bool:
        node = self.search_prefix(word)
        return node is not None and node.leave

    def starts_with(self, prefix: str) -> bool:
        return self.search_prefix(prefix) is not None
