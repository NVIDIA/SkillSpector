import sys
import re

path2 = "src/skillspector/nodes/analyzers/whitespace_padding.py"
with open(path2, "r") as f:
    text2 = f.read()

replacement_rep = """
    runs: list[PaddingRun] = []
    # Fast regex path for character repetition
    import re
    # We look for any character repeated REPEATED_CHAR_THRESHOLD times
    for match in re.finditer(r"(.)\\1{511,}", content, re.DOTALL):
        ch = match.group(1)
        if not is_padding_char(ch):
            runs.append(
                PaddingRun(
                    kind="character",
                    start_offset=match.start(),
                    start_line=content[:match.start()].count("\\n") + 1,
                    length=match.end() - match.start(),
                    followed_by_content=False,
                    summary=summarize_run(match.group(0)[:200]),
                    end_offset=match.end(),
                )
            )
"""
import ast
# Use a simple string search and replace
old_code = """
    runs: list[PaddingRun] = []
    index = 0
    while index < len(content):
        end = index + 1
        while end < len(content) and content[end] == content[index]:
            end += 1
        if end - index >= REPEATED_CHAR_THRESHOLD and not is_padding_char(content[index]):
            runs.append(
                PaddingRun(
                    kind="character",
                    start_offset=index,
                    start_line=content[:index].count("\\n") + 1,
                    length=end - index,
                    followed_by_content=False,
                    summary=summarize_run(content[index : index + 200]),
                    end_offset=end,
                )
            )
        index = end
"""

text2 = text2.replace(old_code.strip('\n'), replacement_rep.strip('\n'))
with open(path2, "w") as f:
    f.write(text2)
