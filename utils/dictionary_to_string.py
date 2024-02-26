from typing import Any, Mapping
from ml_collections import ConfigDict


NEW_LINE = "\n"
TAB = "    "
EMPTY = ""
DICT = "dict"
EQUAL_TO = " = "


def dict_to_str_helper(k: Any, v: Any, escape: Any) -> str:
    if isinstance(v, ConfigDict):
        v = v.to_dict()

    if isinstance(v, Mapping):
        s = (
            f"{escape}{k if k is not None else EMPTY}"
            f"{EQUAL_TO if k is not None else EMPTY}{DICT}({NEW_LINE}"
        )
        for kk, vv in v.items():
            s += f"{dict_to_str_helper(kk, vv, escape + TAB)},{NEW_LINE}"
        s += f"{escape}" + ")"
        return s
    elif isinstance(v, (list, tuple)):
        s = f"{escape}{k}{EQUAL_TO}[{NEW_LINE}"
        for item in v:
            s += f"{dict_to_str_helper(None, item, escape + TAB)},{NEW_LINE}"
        s += f"{escape}]"
        return s
    else:
        if k is None:
            return f"{escape}{v}"
        else:
            return f"{escape}{k}={v}"


def dict_to_str(d: Mapping[Any, Any]) -> str:
    """
    Dyanmically converts a dictionary to a formatted string representation.
    """

    s = f"dict({NEW_LINE}"
    for k, v in d.items():
        s += f"{dict_to_str_helper(k, v, TAB)},{NEW_LINE}"
    s += ")"
    return s