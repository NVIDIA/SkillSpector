import sys
import re

# Fix prompt injection
path = "src/skillspector/nodes/analyzers/static_patterns_prompt_injection.py"
with open(path, "r") as f:
    text = f.read()

replacement = """_TAG_BLOCK_RE = re.compile(rf"[{chr(_TAG_BLOCK[0])}-{chr(_TAG_BLOCK[1])}]")

def _first_smuggled_tag_offset(content: str) -> int | None:
    \"\"\"Return the char offset of the first Unicode Tag character that is *not*
    part of a well-formed emoji tag sequence, or ``None`` if there is none.\"\"\"
    if not _TAG_BLOCK_RE.search(content):
        return None"""
text = re.sub(r'def _first_smuggled_tag_offset.*?if not any\(_TAG_BLOCK\[0\].*?return None', replacement, text, flags=re.DOTALL)
with open(path, "w") as f:
    f.write(text)


# Fix whitespace padding
path2 = "src/skillspector/nodes/analyzers/whitespace_padding.py"
with open(path2, "r") as f:
    text2 = f.read()

# 1. fix _is_blank_line
text2 = text2.replace(
    '    return all(is_padding_char(ch) for ch in line)',
    '    if not line or line.isspace():\n        return True\n    return all(is_padding_char(ch) for ch in line.strip(" \\t\\n\\r\\v\\f"))'
)

# 2. fix padding_bytes ratio check
replacement_ratio = """
        # Fast path for spaces (most common)
        spaces = content.count(" ")
        newlines = content.count("\\n")
        carriage = content.count("\\r")
        tabs = content.count("\\t")
        fast_bytes = spaces + newlines + carriage + tabs
        # Only do the slow check if necessary (or just check the remaining string)
        remaining = content.replace(" ", "").replace("\\n", "").replace("\\r", "").replace("\\t", "")
        padding_bytes = fast_bytes + sum(len(ch.encode("utf-8")) for ch in remaining if is_padding_char(ch))
"""
text2 = re.sub(r'padding_bytes = sum\(len\(ch\.encode\("utf-8"\)\) for ch in content if is_padding_char\(ch\)\)', replacement_ratio.strip(), text2)

# 3. fix _detect_repetition
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
text2 = re.sub(r'runs: list\[PaddingRun\] = \[\]\s+index = 0\s+while index < len\(content\):\s+end = index \+ 1\s+while end < len\(content\) and content\[end\] == content\[index\]:\s+end \+= 1\s+if end - index >= REPEATED_CHAR_THRESHOLD and not is_padding_char\(content\[index\]\):\s+runs\.append\(\s+PaddingRun\(\s+kind="character",\s+start_offset=index,\s+start_line=content\[:index\]\.count\("\\n"\) \+ 1,\s+length=end - index,\s+followed_by_content=False,\s+summary=summarize_run\(content\[index : index \+ 200\]\),\s+end_offset=end,\s+\)\s+\)\s+index = end', replacement_rep.strip(), text2)

with open(path2, "w") as f:
    f.write(text2)

