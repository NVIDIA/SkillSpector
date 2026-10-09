import re
with open("src/skillspector/nodes/analyzers/whitespace_padding.py", "r") as f:
    text = f.read()

old = """    index = 0
    while index < len(content):
        end = index + 1
        while end < len(content) and content[end] == content[index]:
            end += 1
        if end - index >= REPEATED_CHAR_THRESHOLD and not is_padding_char(content[index]):
            runs.append(
                PaddingRun(
                    kind="repetition",
                    start_offset=index,
                    start_line=content[:index].count("\\n") + 1,
                    length=end - index,
                    followed_by_content=end < len(content),
                    summary=f"repeated U+{ord(content[index]):04X} x{end - index}",
                    end_offset=end,
                )
            )
        index = end"""

new = """    import re
    for match in re.finditer(r"(.)\\1{511,}", content, re.DOTALL):
        if not is_padding_char(match.group(1)):
            runs.append(
                PaddingRun(
                    kind="repetition",
                    start_offset=match.start(),
                    start_line=content[:match.start()].count("\\n") + 1,
                    length=match.end() - match.start(),
                    followed_by_content=match.end() < len(content),
                    summary=f"repeated U+{ord(match.group(1)):04X} x{match.end() - match.start()}",
                    end_offset=match.end(),
                )
            )"""
text = text.replace(old, new)
with open("src/skillspector/nodes/analyzers/whitespace_padding.py", "w") as f:
    f.write(text)
