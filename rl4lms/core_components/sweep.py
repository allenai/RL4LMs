from typing import Dict, Any, List
import json
import hashlib
from itertools import product


def get_dict_obj(keys: List, values: List) -> Dict:
    dict = {}
    for key, value in zip(keys, values):
        dict[key] = value
    return dict


def find_products(splits_by_keys: Dict) -> List[Dict]:
    values = list(splits_by_keys.values())
    keys = list(splits_by_keys.keys())
    if len(values) == 1:
        dict_objs = [get_dict_obj(keys, [value]) for value in values[0]]
    else:
        product_values = product(*values)
        dict_objs = [get_dict_obj(keys, value) for value in product_values]
    return dict_objs


def to_expand(obj: Any) -> bool:
    expand = True if isinstance(obj, dict) and obj.get(
        "expand", False) else False
    return expand


def split_config(obj: Dict) -> List[Dict]:
    """
    Recursively splits the given object
    """
    if not isinstance(obj, dict):
        return obj

    # it is a dict and further split
    splits_by_key = {}
    for key, child_obj in obj.items():
        if to_expand(child_obj):
            all_splits = []
            for item in child_obj["values"]:
                splits = split_config(item)
                if isinstance(splits, list):
                    all_splits.extend(splits)
                else:
                    all_splits.append(splits)
            splits_by_key[key] = all_splits

        elif isinstance(child_obj, dict):  # anoter dict, which needs to be expanded
            splits_by_key[key] = split_config(child_obj)
        else:  # others which need not be expanded
            splits_by_key[key] = [child_obj]

    # here, find cartesian
    configs = find_products(splits_by_key)

    return configs


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


if __name__ == "__main__":
    config = {
        "param_1": {
            "expand": True,
            "values": [1, 2]
        },
        "param_3": {
            "param_3_2":  {
                "expand": False,
                "values": [3, 4]
            },
            "param_3_3": 5
        }
    }

    configs = split_config(config)
    print(f" Total configs found: {len(configs)}")
    for config in configs:
        print(config)
        print(dict_hash(config))
