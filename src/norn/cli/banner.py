from __future__ import annotations

from rich.console import Console
from rich.text import Text

ICE_LIGHT = "#9ECFE0"
ICE_PRIMARY = "#7FB8D6"
ICE_DEEP = "#3A6B8A"
SHADOW = "#142535"
GOLD = "#E8C77A"
TEXT_DIM = "#5A6B7A"

_WORDMARK_BLOCK = [
    "███▄  ██   ██████   ██▀▀▄   ███▄  ██",
    "████▄ ██  ██    ██  ██  █   ████▄ ██",
    "██ ████   ██    ██  ████    ██ ████ ",
    "██  ███   ██    ██  ██ ██   ██  ███ ",
    "██   ██    ██████   ██  ██  ██   ██ ",
]

_GRADIENT = [ICE_LIGHT, ICE_LIGHT, ICE_PRIMARY, ICE_DEEP, ICE_DEEP]

_THREAD_UNIT = "·───"
_TAGLINE = "weaves your destiny"

_SHADOW_DX = 2
_SHADOW_DY = 1


def _composite(lines: list[str]) -> list[list[tuple[str, str | None]]]:
    height = len(lines)
    width = max(len(l) for l in lines)
    padded = [l.ljust(width) for l in lines]
    total_h = height + _SHADOW_DY
    total_w = width + _SHADOW_DX
    result: list[list[tuple[str, str | None]]] = []
    for ri in range(total_h):
        row: list[tuple[str, str | None]] = []
        for ci in range(total_w):
            if ri < height and ci < width and padded[ri][ci] != " ":
                row.append((padded[ri][ci], _GRADIENT[ri]))
            else:
                src_ri = ri - _SHADOW_DY
                src_ci = ci - _SHADOW_DX
                if (
                    0 <= src_ri < height
                    and 0 <= src_ci < width
                    and padded[src_ri][src_ci] != " "
                ):
                    row.append((padded[src_ri][src_ci], SHADOW))
                else:
                    row.append((" ", None))
        result.append(row)
    return result


def render_banner() -> str:
    body = "\n".join(_WORDMARK_BLOCK)
    width = len(_WORDMARK_BLOCK[0])
    thread = (_THREAD_UNIT * 15)[:width]
    return f"{body}\n{thread}\n{_TAGLINE.center(width)}"


def print_banner(console: Console) -> None:
    cells = _composite(_WORDMARK_BLOCK)
    total_w = len(cells[0])
    for row in cells:
        text = Text()
        for ch, color in row:
            text.append(ch, style=color or "")
        console.print(text, highlight=False)
    thread = (_THREAD_UNIT * 15)[:total_w]
    console.print(thread, style=GOLD, highlight=False)
    console.print(_TAGLINE.center(total_w), style=f"italic {TEXT_DIM}", highlight=False)
