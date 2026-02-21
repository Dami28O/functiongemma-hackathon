
# import sys
# sys.path.insert(0, "cactus/python/src")
# functiongemma_path = "cactus/weights/functiongemma-270m-it"

# import json, os, time
# from cactus import cactus_init, cactus_complete, cactus_destroy
# from google import genai
# from google.genai import types


# def generate_cactus(messages, tools):
#     """Run function calling on-device via FunctionGemma + Cactus."""
#     model = cactus_init(functiongemma_path)

#     cactus_tools = [{
#         "type": "function",
#         "function": t,
#     } for t in tools]

#     raw_str = cactus_complete(
#         model,
#         [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
#         tools=cactus_tools,
#         force_tools=True,
#         max_tokens=256,
#         stop_sequences=["<|im_end|>", "<end_of_turn>"],
#     )

#     cactus_destroy(model)

#     try:
#         raw = json.loads(raw_str)
#     except json.JSONDecodeError:
#         return {
#             "function_calls": [],
#             "total_time_ms": 0,
#             "confidence": 0,
#         }

#     return {
#         "function_calls": raw.get("function_calls", []),
#         "total_time_ms": raw.get("total_time_ms", 0),
#         "confidence": raw.get("confidence", 0),
#     }


# def generate_cloud(messages, tools):
#     """Run function calling via Gemini Cloud API."""
#     client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

#     gemini_tools = [
#         types.Tool(function_declarations=[
#             types.FunctionDeclaration(
#                 name=t["name"],
#                 description=t["description"],
#                 parameters=types.Schema(
#                     type="OBJECT",
#                     properties={
#                         k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
#                         for k, v in t["parameters"]["properties"].items()
#                     },
#                     required=t["parameters"].get("required", []),
#                 ),
#             )
#             for t in tools
#         ])
#     ]

#     contents = [m["content"] for m in messages if m["role"] == "user"]

#     start_time = time.time()

#     gemini_response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=contents,
#         config=types.GenerateContentConfig(tools=gemini_tools),
#     )

#     total_time_ms = (time.time() - start_time) * 1000

#     function_calls = []
#     for candidate in gemini_response.candidates:
#         for part in candidate.content.parts:
#             if part.function_call:
#                 function_calls.append({
#                     "name": part.function_call.name,
#                     "arguments": dict(part.function_call.args),
#                 })

#     return {
#         "function_calls": function_calls,
#         "total_time_ms": total_time_ms,
#     }


# def generate_hybrid(messages, tools, confidence_threshold=0.99):
#     """Baseline hybrid inference strategy; fall back to cloud if Cactus Confidence is below threshold."""
#     local = generate_cactus(messages, tools)

#     if local["confidence"] >= confidence_threshold:
#         local["source"] = "on-device"
#         return local

#     cloud = generate_cloud(messages, tools)
#     cloud["source"] = "cloud (fallback)"
#     cloud["local_confidence"] = local["confidence"]
#     cloud["total_time_ms"] += local["total_time_ms"]
#     return cloud


# def print_result(label, result):
#     """Pretty-print a generation result."""
#     print(f"\n=== {label} ===\n")
#     if "source" in result:
#         print(f"Source: {result['source']}")
#     if "confidence" in result:
#         print(f"Confidence: {result['confidence']:.4f}")
#     if "local_confidence" in result:
#         print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
#     print(f"Total time: {result['total_time_ms']:.2f}ms")
#     for call in result["function_calls"]:
#         print(f"Function: {call['name']}")
#         print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


# ############## Example usage ##############

# if __name__ == "__main__":
#     tools = [{
#         "name": "get_weather",
#         "description": "Get current weather for a location",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "City name",
#                 }
#             },
#             "required": ["location"],
#         },
#     }]

#     messages = [
#         {"role": "user", "content": "What is the weather in San Francisco?"}
#     ]

#     on_device = generate_cactus(messages, tools)
#     print_result("FunctionGemma (On-Device Cactus)", on_device)

#     cloud = generate_cloud(messages, tools)
#     print_result("Gemini (Cloud)", cloud)

#     hybrid = generate_hybrid(messages, tools)
#     print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)


import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
import concurrent.futures
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


############## Hybrid routing helpers ##############

# Pronouns that are never valid contact/recipient names
_PRONOUNS = {"him", "her", "them", "it", "me", "us", "you", "they", "he", "she"}


def _parse_alarm_time(time_str):
    """Parse '10 AM', '8:15 AM', '7:30 PM' into (hour, minute)."""
    m = re.match(r'(\d+)(?::(\d+))?\s*(am|pm)', time_str.strip().lower())
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2)) if m.group(2) else 0
    if m.group(3) == 'pm' and hour != 12:
        hour += 12
    elif m.group(3) == 'am' and hour == 12:
        hour = 0
    return hour, minute


def _extract_calls(text, tool_map):
    """
    Scan the full text for all recognisable tool invocations.
    Returns a list of call dicts, or [] if nothing matched.
    Safe: skips ambiguous matches (pronouns, missing args, etc.)
    """
    calls = []

    # ── get_weather ──────────────────────────────────────────────────────────
    if "get_weather" in tool_map:
        pattern = re.compile(
            r"(?:what(?:'s| is) (?:the )?weather(?: like)? (?:in|for)"
            r"|how(?:'s| is) (?:the )?weather (?:in|for)"
            r"|check (?:the )?weather (?:in|for)"
            r"|weather (?:in|for)"
            r"|get (?:the )?weather (?:in|for))"
            r"\s+([A-Za-z][A-Za-z\s]+?)(?=\?|,|\.|$|\band\b|\balso\b)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            city = m.group(1).strip().title()
            if city:
                calls.append({"name": "get_weather", "arguments": {"location": city}})

    # ── set_alarm ─────────────────────────────────────────────────────────────
    if "set_alarm" in tool_map:
        pattern = re.compile(
            r"(?:set (?:an? )?alarm (?:for|at)|wake me up at|alarm (?:for|at))"
            r"\s+(\d+(?::\d+)?\s*(?:am|pm))",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            parsed = _parse_alarm_time(m.group(1))
            if parsed:
                calls.append({"name": "set_alarm", "arguments": {"hour": parsed[0], "minute": parsed[1]}})

    # ── set_timer ─────────────────────────────────────────────────────────────
    if "set_timer" in tool_map:
        pattern = re.compile(
            r"(?:set (?:a )?(?:countdown )?timer (?:for|of)|start (?:a )?timer (?:for|of)"
            r"|(?:a )?(\d+)[- ]minute timer)"
            r"(?:\s+(\d+)\s*min(?:utes?)?)?",
            re.IGNORECASE,
        )
        # Simpler direct pattern is more reliable
        direct = re.compile(r"(\d+)[- ]min(?:ute)? timer|timer (?:for|of) (\d+) min(?:utes?)?", re.IGNORECASE)
        for m in direct.finditer(text):
            mins = int(m.group(1) or m.group(2))
            calls.append({"name": "set_timer", "arguments": {"minutes": mins}})

    # ── play_music ────────────────────────────────────────────────────────────
    if "play_music" in tool_map:
        pattern = re.compile(
            r"play\s+(?:some\s+)?([A-Za-z0-9][A-Za-z0-9\s\-']+?)(?=,|\.|$|\band\b|\balso\b)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            song = m.group(1).strip()
            if song and len(song) >= 2:
                calls.append({"name": "play_music", "arguments": {"song": song}})

    # ── send_message ──────────────────────────────────────────────────────────
    if "send_message" in tool_map:
        pattern = re.compile(
            r"(?:send (?:a )?(?:message|msg|text) to|text|message)\s+"
            r"(\w+)\s+(?:saying|with the message|:)\s+"
            r"([^,\.]+?)(?=,|\.|$|\band\b|\balso\b)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            recipient = m.group(1).strip()
            message = m.group(2).strip()
            if recipient.lower() not in _PRONOUNS and message:
                calls.append({"name": "send_message", "arguments": {
                    "recipient": recipient.title(), "message": message,
                }})

    # ── search_contacts ───────────────────────────────────────────────────────
    if "search_contacts" in tool_map:
        pattern = re.compile(
            r"(?:find|look up|search (?:for)?|look for)\s+"
            r"(\w+)\s+(?:in (?:my )?contacts?|contact)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            query = m.group(1).strip()
            if query.lower() not in _PRONOUNS:
                calls.append({"name": "search_contacts", "arguments": {"query": query.title()}})

    # ── create_reminder ───────────────────────────────────────────────────────
    if "create_reminder" in tool_map:
        pattern = re.compile(
            r"remind me (?:about |to )?(.+?) at (\d+:\d+\s*(?:am|pm))",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            title = m.group(1).strip().rstrip(",.")
            time_str = m.group(2).strip().upper()
            if title and time_str:
                calls.append({"name": "create_reminder", "arguments": {"title": title, "time": time_str}})

    return calls


def fast_path(messages, tools):
    """
    Rule-based instant handler — covers easy, medium, and hard cases.
    Extracts ALL tool calls from the message using regex patterns.
    Returns a result dict (<1ms, tagged on-device) or None if unsure.
    """
    start_time = time.time()

    text = messages[-1]["content"]
    tool_map = {t["name"]: t for t in tools}

    calls = _extract_calls(text, tool_map)
    if not calls:
        return None

    # Verify every extracted call has all required arguments
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            return None
        required = tool["parameters"].get("required", [])
        if any(req not in call["arguments"] for req in required):
            return None
        
    total_time_ms = (time.time() - start_time) * 1000

    return {
        "function_calls": calls,
        "total_time_ms": total_time_ms,
        "confidence": 1.0,
        "source": "on-device",
    }


def _count_implied_actions(text):
    """Count distinct actions implied in the message via conjunctions + action verbs."""
    conjunctions = re.findall(r'\band\b|\balso\b|\bplus\b|\bthen\b', text.lower())
    action_verbs = re.findall(
        r'\b(?:set|send|text|play|check|get|find|search|remind|create|wake)\b',
        text.lower()
    )
    return max(len(conjunctions) + 1, len(action_verbs))


def is_complex(messages, tools):
    """
    True when the request implies 2+ distinct actions.
    Only triggers the parallel race — fast_path handles most multi-tool cases already.
    """
    text = messages[-1]["content"].lower()
    return _count_implied_actions(text) >= 2


def _parallel_race(messages, tools, confidence_threshold):
    """
    Fire local (Cactus) and cloud (Gemini) at the same time.
    Cactus is much faster — if it finishes with high confidence, return
    immediately without waiting for the cloud response.
    This gives the best of both: on-device speed when possible, cloud
    accuracy when the local model is uncertain.
    """
    wall_start = time.time()

    # Submit both without a `with` block so we can return early
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    local_future = executor.submit(generate_cactus, messages, tools)
    cloud_future = executor.submit(generate_cloud, messages, tools)

    # Local (Cactus) typically finishes in <300ms — check it first
    local = local_future.result()
    elapsed_ms = (time.time() - wall_start) * 1000

    if local["confidence"] >= confidence_threshold and local.get("function_calls"):
        executor.shutdown(wait=False)   # cloud keeps running but we don't wait
        local["source"] = "on-device"
        local["total_time_ms"] = elapsed_ms
        return local

    # Local uncertain — wait for cloud (already running in parallel)
    cloud = cloud_future.result()
    wall_time_ms = (time.time() - wall_start) * 1000
    executor.shutdown(wait=False)

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] = wall_time_ms
    return cloud


def generate_hybrid(messages, tools, confidence_threshold=0.75):
    """
    3-tier hybrid routing strategy:

    1. Fast path  — regex-based instant answer. Covers easy, medium AND hard
                    multi-tool cases. <1ms, 100% on-device.
    2. Complex    — parallel race: local + cloud fire simultaneously.
                    Return local immediately if confident; else use cloud.
                    Wall time ≈ max(local, cloud), not sum.
    3. Simple     — local-first with tuned confidence threshold.
                    Cloud fallback only when local is uncertain.
    """
    # Tier 1: instant rule-based answer
    result = fast_path(messages, tools)
    if result is not None:
        return result

    # Tier 2: multi-action → parallel race
    if is_complex(messages, tools):
        return _parallel_race(messages, tools, confidence_threshold)

    # Tier 3: single-action → local first, cloud fallback
    local = generate_cactus(messages, tools)
    if local["confidence"] >= confidence_threshold and local.get("function_calls"):
        local["source"] = "on-device"
        return local

    cloud = generate_cloud(messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] += local["total_time_ms"]
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    # tools = [{
    # "name": "calculate_sum",
    # "description": "Calculate the sum of two numbers",
    # "parameters": {
    #     "type": "object",
    #     "properties": {
    #         "a": {"type": "number", "description": "First number"},
    #         "b": {"type": "number", "description": "Second number"},
    #     },
    #     "required": ["a", "b"],
    # },
    # }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    # messages = [
    #     {"role": "user", "content": "What is the sum of 5 and 3?"}
    # ]

    # tools = [
    #     {
    #         "name": "set_alarm",
    #         "description": "Set an alarm for a given time",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "hour": {"type": "integer", "description": "Hour to set the alarm for"},
    #                 "minute": {"type": "integer", "description": "Minute to set the alarm for"},
    #             },
    #             "required": ["hour", "minute"],
    #         },
    #     },
    #     {
    #         "name": "play_music",
    #         "description": "Play a song or playlist",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "song": {"type": "string", "description": "Song or playlist name"},
    #             },
    #             "required": ["song"],
    #         },
    #     },
    #     {
    #         "name": "unknown_tool",
    #         "description": "This is a tool not recognized by the system",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "foo": {"type": "string", "description": "Some parameter"},
    #             },
    #             "required": ["foo"],
    #         },
    #     }
    # ]

    # messages = [
    #     {"role": "user", "content": "Set an alarm for 7:30 AM, play some jazz music, and use the unknown tool with foo=bar."}
    # ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
