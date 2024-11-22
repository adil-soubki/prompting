#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
from src.agents.openai import GPTAgent
from src.core.app import harness
from src.core.context import Context


def main(ctx: Context) -> None:
    ctx.parser.add_argument("--foo", action="store_true")
    args = ctx.parser.parse_args()


if __name__ == "__main__":
    harness(main)
