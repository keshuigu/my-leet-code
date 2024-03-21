import queue
from collections import defaultdict
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
    def __init__(self, val, left=None, right=None):
        if type(val) is list:
            self._init_with_list(val)
            return
        self.val = val
        self.left = left
        self.right = right

    def _init_with_list(self, tree: List[Any]):
        self.val = tree[0]
        p = self
        q = [p]
        level = 0
        while q:
            tmp = q
            q = []
            level += 1
            if 2 ** (level + 1) >= len(tree):
                sub_nodes = tree[2 ** level - 1:]
            else:
                sub_nodes = tree[2 ** level - 1:2 ** (level + 1) - 1]
            index = 0
            if len(sub_nodes) == 0:
                return
            for node in tmp:
                if not node:
                    index += 2
                    continue
                if index >= len(sub_nodes):
                    break
                node.left = TreeNode(sub_nodes[index]) if sub_nodes[index] is not None else None
                q.append(node.left)
                index += 1
                if index >= len(sub_nodes):
                    break
                node.right = TreeNode(sub_nodes[index]) if sub_nodes[index] is not None else None
                q.append(node.right)
                index += 1
                if index >= len(sub_nodes):
                    break
            if 2 ** (level + 1) >= len(tree):
                return

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


class PriorityQueue:
    # 最大值在最小的位置
    def __init__(self, capacity: int = 65535) -> None:
        self.capacity = capacity
        self.queue: List[int] = [0] * (capacity + 1)
        self.queue[0] = 0xffffffff  # 哨兵
        self.size: int = 0

    def empty(self) -> bool:
        if self.size == 0:
            return True
        else:
            return False

    def full(self) -> bool:
        if self.size == self.capacity:
            return True
        else:
            return False

    def put(self, val: int) -> None:
        if self.full():
            return
        i = self.size + 1
        self.size += 1
        while i > 0:  # 有哨兵，可以循环到0
            if self.queue[i // 2] < val:
                self.queue[i] = self.queue[i // 2]
                i = i // 2
            else:
                break
        self.queue[i] = val

    def delete(self) -> Union[None, int]:
        if self.empty():
            return None
        max_val = self.queue[1]
        # 取最小值进行下沉
        tmp = self.queue[self.size]
        self.size -= 1
        parent = 1
        while parent * 2 <= self.size:  # 注意循环边界，size已经-1
            child = parent * 2
            # 选大的一边
            if child != self.size and self.queue[child] < self.queue[child + 1]:
                child += 1
            if tmp >= self.queue[child]:
                break
            else:
                self.queue[parent] = self.queue[child]
            parent = child
        self.queue[parent] = tmp
        return max_val

    def top(self) -> Union[None, int]:
        if self.empty():
            return None
        else:
            return self.queue[1]


class StackWithQueue:
    def __init__(self):
        self.items = []
        self.top_p = -1

    def push(self, x: int) -> None:
        self.items.append(x)
        self.top_p += 1

    def pop(self) -> int:
        tmp_top = self.top_p
        tmp = 0
        while tmp_top >= 0:
            tmp = self.items[0]
            self.items.remove(tmp)
            if tmp_top != 0:
                self.items.append(tmp)
            tmp_top -= 1
        self.top_p -= 1
        return tmp

    def top(self) -> int:
        tmp_top = self.top_p
        tmp = 0
        while tmp_top >= 0:
            tmp = self.items[0]
            self.items.remove(tmp)
            self.items.append(tmp)
            tmp_top -= 1
        return tmp

    def empty(self) -> bool:
        return self.top_p == -1


class QueueWithStack:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
        self.size = 0

    def push(self, x: int) -> None:
        self.stack1.append(x)
        self.size += 1

    def pop(self) -> int:
        while len(self.stack1) > 1:
            self.stack2.append(self.stack1.pop())
        tmp = self.stack1.pop()
        self.size -= 1
        while len(self.stack2) > 0:
            self.stack1.append(self.stack2.pop())
        return tmp

    def peek(self) -> int:
        while len(self.stack1) > 1:
            self.stack2.append(self.stack1.pop())
        tmp = self.stack1.pop()
        self.stack2.append(tmp)
        while len(self.stack2) > 0:
            self.stack1.append(self.stack2.pop())
        return tmp

    def empty(self) -> bool:
        return self.size == 0


class QueueWithStack2:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []
        self.size = 0

    def push(self, x: int) -> None:
        self.stack1.append(x)
        self.size += 1

    def pop(self) -> int:
        if len(self.stack2) == 0:
            while len(self.stack1) > 0:
                self.stack2.append(self.stack1.pop())
        self.size -= 1
        return self.stack2.pop()

    def peek(self) -> int:
        if len(self.stack2) == 0:
            while len(self.stack1) > 0:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self) -> bool:
        return self.size == 0


class Node:
    def __init__(self, val):
        if type(val) is list:
            self._init_with_list(val)
            return
        self.val = val
        self.children = []

    def _init_with_list(self, vals: List[Any]):
        self.val = vals[0]
        self.children = []
        q = [self]
        index = 2
        while q:
            tmp = q
            q = []
            for node in tmp:
                while vals[index] is not None:
                    cur = Node(val=vals[index])
                    q.append(cur)
                    node.children.append(cur)
                    index += 1
                    if index == len(vals):
                        return
                index += 1
                if index == len(vals):
                    return

    def __str__(self):
        q = [self]
        res = ''
        while q:
            tmp = q
            q = []
            cur = ''
            for node in tmp:
                cur += str(node.val)
                for child in node.children:
                    q.append(child)
            res += '[' + cur + '],'
        return res


class TireOf3045:
    def __init__(self):
        self.son = dict()  # key 是 pair value 是Node
        self.cnt = 0  # 以该节点结尾的字符串的出现次数


class Fenwick:
    __slots__ = 'tree'

    def __init__(self, n: int):
        self.tree = [0] * n

    def add(self, i: int, v: int) -> None:
        """
        把下标为i的元素增加v
        """
        while i < len(self.tree):
            self.tree[i] += v
            i += i & -i  # i加上low bit

    def pre(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.tree[i]
            i &= i - 1  # i减去low bit <=> i -= i & (-i)
        return res


class NumArray:
    __slots__ = "sums"

    def __init__(self, nums: List[int]):
        self.sums = [0] * (len(nums) + 1)
        for i, num in enumerate(nums):
            self.sums[i + 1] = self.sums[i] + num

    def sumRange(self, left: int, right: int):
        return self.sums[right + 1] - self.sums[left]


class FindElements:
    def __init__(self, root: Optional[TreeNode]):
        self.root = root
        root.val = 0
        s = [self.root]
        p = self.root
        while s:
            while p:
                s.append(p)
                if p.left:
                    p.left.val = p.val * 2 + 1
                    p = p.left
                else:
                    p = None
            p = s.pop()
            if p.right:
                p.right.val = p.val * 2 + 2
                p = p.right
            else:
                p = None
        return

    def find(self, target: int) -> bool:
        order = []
        while target > 0:
            if target % 2 == 1:
                order.append(0)
            else:
                order.append(1)
            target = (target - 1) // 2
        order.reverse()
        p = self.root
        for o in order:
            if o:
                p = p.right
                if not p:
                    return False
            else:
                p = p.left
                if not p:
                    return False
        return True

    def find_2(self, target: int) -> bool:
        # 位运算,所有节点值加1,会形成规律
        # root = 1
        # root.left = 10 root.right = 11
        # 100 101 110 111
        # 0走左 1走右
        target += 1
        cur = self.root
        for i in range(target.bit_length() - 2, -1, -1):  # 从次高位开始枚举
            o = 1 & (target >> i)
            if o:
                cur = cur.left
            else:
                cur = cur.right
            if not cur:
                return False
        return True


class FrequencyTracker:

    def __init__(self):
        self.table1 = defaultdict(int)
        self.table2 = defaultdict(int)

    def add(self, number: int) -> None:
        cur = self.table1[number]
        self.table1[number] += 1
        if cur > 0:
            self.table2[cur] -= 1
        self.table2[cur + 1] += 1

    def deleteOne(self, number: int) -> None:
        if self.table1[number] == 0:
            return
        cur = self.table1[number]
        self.table1[number] -= 1
        self.table2[cur] -= 1
        if cur > 1:
            self.table2[cur - 1] += 1

    def hasFrequency(self, frequency: int) -> bool:
        return frequency in self.table2 and self.table2[frequency] != 0
