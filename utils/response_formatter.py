def format_response(raw_text):
    lines = raw_text.strip().split("\n")
    return "\n".join(f"- {line}" for line in lines if line.strip())
