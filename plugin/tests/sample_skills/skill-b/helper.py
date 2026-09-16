def helper():
    return "hello from skill-b"


# No secrets, no network, just local
with open("data.txt", errors="ignore") as f:
    try:
        x = f.read()
    except:
        x = ""
