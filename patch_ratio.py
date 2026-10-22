with open("src/skillspector/nodes/analyzers/whitespace_padding.py", "r") as f:
    text = f.read()

old = 'padding_bytes = sum(len(ch.encode("utf-8")) for ch in content if is_padding_char(ch))'
new = """        # Fast path for spaces
        spaces = content.count(" ")
        newlines = content.count("\\n")
        carriage = content.count("\\r")
        tabs = content.count("\\t")
        fast_bytes = spaces + newlines + carriage + tabs
        remaining = content.replace(" ", "").replace("\\n", "").replace("\\r", "").replace("\\t", "")
        padding_bytes = fast_bytes + sum(len(ch.encode("utf-8")) for ch in remaining if is_padding_char(ch))"""

text = text.replace(old, new)
with open("src/skillspector/nodes/analyzers/whitespace_padding.py", "w") as f:
    f.write(text)
