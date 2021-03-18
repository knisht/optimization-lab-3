from typing import List


def format_list(list_: List[float]) -> str:
    return '[' + ' '.join(list(map(lambda s: f'{s:.2f}', list_))) + ']'
