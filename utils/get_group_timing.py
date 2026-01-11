import json


def get_group_timing(data, group_text):
    """
    data : dict
    group_text : str, ex. "Hello, how are you?"
    """
    words = data["words_timings"]
    group_tokens = group_text.strip().split()

    for i in range(len(words)):
        match = True
        for j, token in enumerate(group_tokens):
            if i + j >= len(words):
                match = False
                break
            if words[i + j]["word"].strip(",.?!") != token.strip(",.?!"):
                match = False
                break

        if match:
            start = words[i].get("startTime")
            end = words[i + len(group_tokens) - 1].get("endTime")
            return {"group": group_text, "startTime": start, "endTime": end}

    return None
