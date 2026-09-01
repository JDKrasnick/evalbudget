"""Tiny deterministic model command used by the quickstart."""

import re
import sys

prompt = sys.stdin.read()
numbers = [int(value) for value in re.findall(r"-?\d+", prompt)]
print(sum(numbers))
