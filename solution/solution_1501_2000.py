from typing import *
from .data_struct import *


def solution_1672(accounts: List[List[int]]) -> int:
    max_account = 0
    for account in accounts:
        one_total = 0
        for bank in account:
            one_total = one_total + bank
        if one_total > max_account:
            max_account = one_total
    return max_account
