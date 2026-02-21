"""Quick offline test for fast_path regex patterns — no Cactus or Gemini needed."""
import re

_PRONOUNS = {"him", "her", "them", "it", "me", "us", "you", "they", "he", "she"}


def _parse_alarm_time(time_str):
    m = re.match(r"(\d+)(?::(\d+))?\s*(am|pm)", time_str.strip().lower())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    if m.group(3) == "pm" and hour != 12:
        hour += 12
    elif m.group(3) == "am" and hour == 12:
        hour = 0
    return hour, minute


def _extract_calls(text, tool_map):
    calls = []

    if "get_weather" in tool_map:
        p = re.compile(
            r"(?:what(?:'s| is) (?:the )?weather(?: like)? (?:in|for)"
            r"|how(?:'s| is) (?:the )?weather (?:in|for)"
            r"|check (?:the )?weather (?:in|for)"
            r"|weather (?:in|for)"
            r"|get (?:the )?weather (?:in|for))"
            r"\s+([A-Za-z][A-Za-z\s]+?)(?=\?|,|\.|\band\b|\balso\b|$)",
            re.IGNORECASE,
        )
        for m in p.finditer(text):
            city = m.group(1).strip().title()
            if city:
                calls.append({"name": "get_weather", "arguments": {"location": city}})

    if "set_alarm" in tool_map:
        p = re.compile(
            r"(?:set (?:an? )?alarm (?:for|at)|wake me up at|alarm (?:for|at))"
            r"\s+(\d+(?::\d+)?\s*(?:am|pm))",
            re.IGNORECASE,
        )
        for m in p.finditer(text):
            parsed = _parse_alarm_time(m.group(1))
            if parsed:
                calls.append({"name": "set_alarm", "arguments": {"hour": parsed[0], "minute": parsed[1]}})

    if "set_timer" in tool_map:
        direct = re.compile(
            r"(\d+)[- ]min(?:ute)? timer"
            r"|timer (?:for|of) (\d+) min(?:utes?)?"
            r"|set (?:a )?(?:countdown )?timer (?:for|of) (\d+) min(?:utes?)?"
            r"|start (?:a )?timer (?:for|of) (\d+) min(?:utes?)?",
            re.IGNORECASE,
        )
        for m in direct.finditer(text):
            mins = int(next(g for g in m.groups() if g is not None))
            calls.append({"name": "set_timer", "arguments": {"minutes": mins}})

    if "play_music" in tool_map:
        p = re.compile(
            r"play\s+(?:some\s+)?([A-Za-z0-9][A-Za-z0-9\s\-']+?)(?=,|\.|\band\b|\balso\b|$)",
            re.IGNORECASE,
        )
        for m in p.finditer(text):
            song = m.group(1).strip()
            if song and len(song) >= 2:
                calls.append({"name": "play_music", "arguments": {"song": song}})

    if "send_message" in tool_map:
        p = re.compile(
            r"(?:send (?:a )?(?:message|msg|text) to|text|message)\s+"
            r"(\w+)\s+(?:saying|with the message|:)\s+"
            r"([^,\.]+?)(?=,|\.|\band\b|\balso\b|$)",
            re.IGNORECASE,
        )
        for m in p.finditer(text):
            rec = m.group(1).strip()
            msg = m.group(2).strip()
            if rec.lower() not in _PRONOUNS and msg:
                calls.append({"name": "send_message", "arguments": {"recipient": rec.title(), "message": msg}})

    if "search_contacts" in tool_map:
        p = re.compile(
            r"(?:find|look up|search (?:for)?|look for)\s+"
            r"(\w+)\s+(?:in (?:my )?contacts?|contact)",
            re.IGNORECASE,
        )
        for m in p.finditer(text):
            q = m.group(1).strip()
            if q.lower() not in _PRONOUNS:
                calls.append({"name": "search_contacts", "arguments": {"query": q.title()}})

    if "create_reminder" in tool_map:
        p = re.compile(
            r"remind me (?:about |to )?(.+?) at (\d+:\d+\s*(?:am|pm))",
            re.IGNORECASE,
        )
        _leading_articles = re.compile(r'^(?:the|a|an)\s+', re.IGNORECASE)
        for m in p.finditer(text):
            title = m.group(1).strip().rstrip(",. ")
            title = _leading_articles.sub('', title)
            t = m.group(2).strip().upper()
            if title and t:
                calls.append({"name": "create_reminder", "arguments": {"title": title, "time": t}})

    return calls


TESTS = [
    # Easy
    ("What is the weather in San Francisco?", ["get_weather"], [{"name": "get_weather", "arguments": {"location": "San Francisco"}}]),
    ("Set an alarm for 10 AM.", ["set_alarm"], [{"name": "set_alarm", "arguments": {"hour": 10, "minute": 0}}]),
    ("Send a message to Alice saying good morning.", ["send_message"], [{"name": "send_message", "arguments": {"recipient": "Alice", "message": "good morning"}}]),
    ("What's the weather like in London?", ["get_weather"], [{"name": "get_weather", "arguments": {"location": "London"}}]),
    ("Wake me up at 6 AM.", ["set_alarm"], [{"name": "set_alarm", "arguments": {"hour": 6, "minute": 0}}]),
    ("Play Bohemian Rhapsody.", ["play_music"], [{"name": "play_music", "arguments": {"song": "Bohemian Rhapsody"}}]),
    ("Set a timer for 5 minutes.", ["set_timer"], [{"name": "set_timer", "arguments": {"minutes": 5}}]),
    ("Remind me about the meeting at 3:00 PM.", ["create_reminder"], [{"name": "create_reminder", "arguments": {"title": "meeting", "time": "3:00 PM"}}]),
    ("Find Bob in my contacts.", ["search_contacts"], [{"name": "search_contacts", "arguments": {"query": "Bob"}}]),
    ("How's the weather in Paris?", ["get_weather"], [{"name": "get_weather", "arguments": {"location": "Paris"}}]),
    # Medium
    ("Send a message to John saying hello.", ["get_weather", "send_message", "set_alarm"], [{"name": "send_message", "arguments": {"recipient": "John", "message": "hello"}}]),
    ("What's the weather in Tokyo?", ["get_weather", "send_message"], [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]),
    ("Set an alarm for 8:15 AM.", ["send_message", "set_alarm", "get_weather"], [{"name": "set_alarm", "arguments": {"hour": 8, "minute": 15}}]),
    ("Play some jazz music.", ["set_alarm", "play_music", "get_weather"], [{"name": "play_music", "arguments": {"song": "jazz music"}}]),
    ("Set a timer for 10 minutes.", ["set_alarm", "set_timer", "play_music"], [{"name": "set_timer", "arguments": {"minutes": 10}}]),
    ("Look up Sarah in my contacts.", ["send_message", "get_weather", "search_contacts", "set_alarm"], [{"name": "search_contacts", "arguments": {"query": "Sarah"}}]),
    ("Text Dave saying I'll be late.", ["get_weather", "set_timer", "send_message", "play_music"], [{"name": "send_message", "arguments": {"recipient": "Dave", "message": "I'll be late"}}]),
    ("Set an alarm for 9 AM.", ["send_message", "get_weather", "play_music", "set_timer", "set_alarm"], [{"name": "set_alarm", "arguments": {"hour": 9, "minute": 0}}]),
    # Hard
    ("Send a message to Bob saying hi and get the weather in London.", ["get_weather", "send_message", "set_alarm"], [{"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}}, {"name": "get_weather", "arguments": {"location": "London"}}]),
    ("Set an alarm for 7:30 AM and check the weather in New York.", ["get_weather", "set_alarm", "send_message"], [{"name": "set_alarm", "arguments": {"hour": 7, "minute": 30}}, {"name": "get_weather", "arguments": {"location": "New York"}}]),
    ("Set a timer for 20 minutes and play lo-fi beats.", ["set_timer", "play_music", "get_weather", "set_alarm"], [{"name": "set_timer", "arguments": {"minutes": 20}}, {"name": "play_music", "arguments": {"song": "lo-fi beats"}}]),
    ("Check the weather in Miami and play summer hits.", ["get_weather", "play_music", "set_timer", "send_message"], [{"name": "get_weather", "arguments": {"location": "Miami"}}, {"name": "play_music", "arguments": {"song": "summer hits"}}]),
    ("Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.", ["send_message", "get_weather", "set_alarm", "play_music", "set_timer"], [{"name": "send_message", "arguments": {"recipient": "Emma", "message": "good night"}}, {"name": "get_weather", "arguments": {"location": "Chicago"}}, {"name": "set_alarm", "arguments": {"hour": 5, "minute": 0}}]),
    ("Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.", ["set_timer", "play_music", "create_reminder", "get_weather", "send_message"], [{"name": "set_timer", "arguments": {"minutes": 15}}, {"name": "play_music", "arguments": {"song": "classical music"}}, {"name": "create_reminder", "arguments": {"title": "stretch", "time": "4:00 PM"}}]),
]


def _matches(call, exp):
    if call["name"] != exp["name"]:
        return False
    return all(
        str(call["arguments"].get(k, "")).lower().strip() == str(v).lower().strip()
        for k, v in exp["arguments"].items()
    )


passed = 0
failed = []
for text, tool_names, expected in TESTS:
    tool_map = {n: {"name": n, "parameters": {"required": []}} for n in tool_names}
    calls = _extract_calls(text, tool_map)
    ok = True
    for exp in expected:
        if not any(_matches(c, exp) for c in calls):
            ok = False
            failed.append(f"  FAIL  [{text[:60]}]\n        expected {exp['name']}({exp['arguments']})\n        got      {calls}")
    if ok:
        passed += 1

print(f"\nfast_path: {passed}/{len(TESTS)} passed\n")
for f in failed:
    print(f)
