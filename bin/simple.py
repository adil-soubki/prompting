#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
from src.agents.openai import GPTAgent
from src.core.app import harness
from src.core.context import Context


def main(ctx: Context) -> None:
    args = ctx.parser.parse_args()
    gpt = GPTAgent(
        "gpt-4o-mini", system_prompt="Answer everything incorrectly", temperature=0
    )
    ctx.log.info(gpt.generate("What is the capital of Burkina Faso?"))


if __name__ == "__main__":
    harness(main)
