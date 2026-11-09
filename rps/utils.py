from pathlib import Path


def list_from_file(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [line.rstrip('\n\r') for line in stream]


def is_type_list(items, item_type):
    return isinstance(items, list) and all(isinstance(item, item_type) for item in items)
