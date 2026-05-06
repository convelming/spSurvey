"""旧入口兼容文件。

历史 PyCharm run configuration 如果仍指向本文件，可以继续运行；
实际逻辑已经迁移到 `gz_pt_share_selfattention_debug_flow.py`。
"""

from __future__ import annotations

from gz_pt_share_selfattention_debug_flow import build_arg_parser, run


if __name__ == '__main__':
    parser = build_arg_parser()
    run(parser.parse_args())
