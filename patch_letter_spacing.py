import re
with open("src/skillspector/artifacts.py", "r") as f:
    text = f.read()

old = """    offset = 0
    while offset < len(text):
        if check_runtime is not None and offset % 4096 == 0:
            check_runtime()
        if not text[offset].isalpha() or (offset > 0 and text[offset - 1].isalpha()):
            offset += 1
            continue

        run_start = offset"""
new = """    offset = 0
    # Fast forward to next potential start (a letter not preceded by a letter)
    import re
    # We just need to find boundaries. A simpler way is to just advance `offset` 
    # to the next `isalpha()` that is preceded by `not isalpha()`.
    # But for a 5MB string of spaces, `offset += 1` in python is too slow.
    # We can use regex to find the next letter!
    letter_pattern = re.compile(r'[^\W\d_]', re.UNICODE)
    
    while offset < len(text):
        if check_runtime is not None:
            check_runtime()
            
        match = letter_pattern.search(text, offset)
        if not match:
            break
            
        offset = match.start()
        if offset > 0 and text[offset - 1].isalpha():
            offset += 1
            continue
            
        run_start = offset"""

text = text.replace(old, new)
with open("src/skillspector/artifacts.py", "w") as f:
    f.write(text)
