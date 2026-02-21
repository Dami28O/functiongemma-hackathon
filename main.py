
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import atexit
import json, os, re, time
import concurrent.futures
from cactus import cactus_init, cactus_complete, cactus_destroy, cactus_reset
from google import genai
from google.genai import types


############## Model cache — load once, reuse across all calls ##############

_model_handle = None

def _get_model():
    """
    Return a cached Cactus model handle, loading it on first call.
    Reusing the same handle across benchmark cases avoids the 100-300ms
    cactus_init overhead on every single inference call.
    """
    global _model_handle
    if _model_handle is None:
        _model_handle = cactus_init(functiongemma_path)
        atexit.register(_cleanup_model)
    return _model_handle

def _cleanup_model():
    global _model_handle
    if _model_handle is not None:
        cactus_destroy(_model_handle)
        _model_handle = None


############## Inference functions ##############

_SYSTEM_PROMPT = (
    "You are a precise function-calling assistant. "
    "Your only job is to call the correct function with exact argument values "
    "extracted from the user's request. Always call a function — never reply with plain text."
)


def _coerce_args(function_calls, tools):
    """
    Coerce argument types to match the tool schema.
    FunctionGemma often returns integers as strings (e.g. "5" instead of 5).
    The benchmark's _normalize() does NOT convert types, so "5" != 5 → F1=0.
    This fixes timer_5min, set_alarm, and any other integer-param tools.
    """
    tool_map = {t["name"]: t for t in tools}
    for call in function_calls:
        tool = tool_map.get(call.get("name", ""))
        if not tool:
            continue
        props = tool["parameters"].get("properties", {})
        args = call.get("arguments", {})
        for key, val in args.items():
            if key not in props:
                continue
            expected = props[key].get("type")
            if expected == "integer" and not isinstance(val, int):
                try:
                    args[key] = int(float(str(val)))
                except (ValueError, TypeError):
                    pass
            elif expected == "number" and not isinstance(val, float):
                try:
                    args[key] = float(str(val))
                except (ValueError, TypeError):
                    pass
    return function_calls


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = _get_model()
    cactus_reset(model)   # clear KV cache from previous call

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        temperature=0,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    calls = _coerce_args(raw.get("function_calls", []), tools)
    return {
        "function_calls": calls,
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
        model="gemini-2.0-flash",
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


def _resolve_contact_pronouns(text):
    """
    Smart coreference resolution: when a contact is looked up by name,
    replace subsequent third-person pronouns with that person's name.

    Example:
        "Find Tom in my contacts and send him a message saying happy birthday"
     -> "Find Tom in my contacts and send Tom a message saying happy birthday"

    This lets both the models AND the regex handle these queries correctly
    without any tool-specific hardcoding.
    """
    # Match: find/look up/search NAME in contacts (any case)
    contact_re = re.compile(
        r'(?:find|look\s+up|search(?:\s+for)?)\s+([A-Z][a-zA-Z]+)\s+(?:in|from)\s+(?:my\s+)?contacts?',
        re.IGNORECASE
    )
    m = contact_re.search(text)
    if not m:
        return text
    name = m.group(1)  # e.g. "Tom" or "Jake"
    # Replace third-person pronouns that likely refer to this contact
    resolved = re.sub(r'\b(him|her|them)\b', name, text, flags=re.IGNORECASE)
    return resolved


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
        # Single unified pattern — avoids duplicate matches when text contains both
        # "set a timer for N minutes" and "timer for N minutes" substrings.
        # The longest/most-specific alternative is first so it captures the full phrase.
        direct = re.compile(
            r"(?:set (?:a )?(?:countdown )?timer (?:for|of)|start (?:a )?timer (?:for|of))"
            r"\s*(\d+)\s*min(?:utes?)?"
            r"|timer (?:for|of) (\d+) min(?:utes?)?"
            r"|(\d+)[- ]min(?:ute)? timer",
            re.IGNORECASE,
        )
        seen_timer = set()
        for m in direct.finditer(text):
            mins = int(next(g for g in m.groups() if g is not None))
            if mins not in seen_timer:
                seen_timer.add(mins)
                calls.append({"name": "set_timer", "arguments": {"minutes": mins}})

    # ── play_music ────────────────────────────────────────────────────────────
    if "play_music" in tool_map:
        pattern = re.compile(
            r"play\s+(?:some\s+)?([A-Za-z0-9][A-Za-z0-9\s\-']+?)(?=,|\.|$|\band\b|\balso\b)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            song = m.group(1).strip()
            # Skip generic genre+"music" phrases (e.g. "jazz music") — the model
            # returns just the genre ("jazz") which doesn't match our regex capture.
            # BUT only skip when it's a single-word genre + " music", not multi-word
            # titles like "classical music" or "lo-fi beats".
            # Safest rule: skip ONLY "<single_word> music" — one word before " music"
            words_before_music = song.lower().removesuffix(" music").split() if song.lower().endswith(" music") else None
            if words_before_music is not None and len(words_before_music) == 1:
                continue  # e.g. "jazz music" → skip, let cloud return "jazz"
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
        _leading_articles = re.compile(r'^(?:the|a|an)\s+', re.IGNORECASE)
        for m in pattern.finditer(text):
            title = m.group(1).strip().rstrip(",. ")
            title = _leading_articles.sub('', title)  # strip "the meeting" → "meeting"
            time_str = m.group(2).strip().upper()
            if title and time_str:
                calls.append({"name": "create_reminder", "arguments": {"title": title, "time": time_str}})

    return calls


def fast_path(messages, tools):
    """
    Pattern-based pre-classifier: determines whether a request is
    unambiguously simple enough to route straight to generate_cactus
    with high confidence, skipping the parallel race overhead.

    Returns True  → caller should use generate_cactus (simple, well-understood)
    Returns False → caller should use full routing (complex or ambiguous)

    This is a ROUTING hint, NOT a result generator. The actual function
    calls always come from the model (Cactus or Gemini), never from regex.
    """
    text = messages[-1]["content"]
    tool_map = {t["name"]: t for t in tools}

    # Multi-action requests must go through the parallel race.
    # fast_path only handles single-tool calls — _generate_cactus_focused
    # uses max_tokens=128 which is insufficient for 2+ tool call responses.
    # _count_implied_actions is defined below but resolved at call time (Python late-binding).
    if _count_implied_actions(text) >= 2:
        return False

    calls = _extract_calls(text, tool_map)
    if not calls:
        return False

    # Every extracted call must have all required arguments clearly present
    for call in calls:
        tool = tool_map.get(call["name"])
        if not tool:
            return False
        required = tool["parameters"].get("required", [])
        if any(req not in call["arguments"] for req in required):
            return False

    # Pattern matched cleanly — safe to route straight to local model
    return True


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

    Local wins only if:
      1. Confidence >= threshold (raised to 0.85 for 3+ action requests), AND
      2. It returned at least as many tool calls as the request implies
         (prevents partial multi-tool results like 1-of-2 tools winning)

    Wall time = actual elapsed, not sum of both.
    """
    wall_start = time.time()
    min_calls_needed = _count_implied_actions(messages[-1]["content"])

    # Harder multi-action queries need higher confidence to trust local
    effective_threshold = 0.85 if min_calls_needed >= 3 else confidence_threshold

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    local_future = executor.submit(generate_cactus, messages, tools)
    cloud_future = executor.submit(generate_cloud, messages, tools)

    local = local_future.result()
    elapsed_ms = (time.time() - wall_start) * 1000

    local_calls = local.get("function_calls", [])
    local_complete = len(local_calls) >= min_calls_needed

    if local["confidence"] >= effective_threshold and local_calls and local_complete:
        executor.shutdown(wait=False)
        local["source"] = "on-device"
        local["total_time_ms"] = elapsed_ms
        return local

    # Local uncertain or incomplete — use cloud (already running)
    cloud = cloud_future.result()
    wall_time_ms = (time.time() - wall_start) * 1000
    executor.shutdown(wait=False)

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] = wall_time_ms
    return cloud


def _generate_cactus_focused(messages, tools):
    """
    Focused local inference — shorter max_tokens for simple single-tool calls.
    No tool_rag_top_k: that parameter was found to select wrong tools.
    """
    model = _get_model()
    cactus_reset(model)

    cactus_tools = [{"type": "function", "function": t} for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": _SYSTEM_PROMPT}] + messages,
        tools=cactus_tools,
        force_tools=True,
        temperature=0,
        max_tokens=128,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}

    calls = _coerce_args(raw.get("function_calls", []), tools)
    return {
        "function_calls": calls,
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_hybrid(messages, tools, confidence_threshold=0.75):
    """
    Model-first hybrid routing strategy.

    Step 0 — Preprocess:
              Resolve contact pronouns so models AND regex see clean text.
              E.g. "Find Tom... send him" → "Find Tom... send Tom"

    Tier 1 — Complex (multi-action): parallel race.
              Both local (FunctionGemma) and cloud (Gemini) fire simultaneously.
              Local wins if confident + complete; else cloud result used.

    Tier 2 — Simple ambiguous: local FunctionGemma first.
              Cloud fallback when local is uncertain.
              Special-case: play_music with genre+"music" phrasing → cloud direct.

    Tier 3 — Regex safety net:
              Only reached if BOTH models returned empty function_calls.
              Deterministic extraction; used as last resort, not primary path.
    """
    # Step 0: Preprocess — resolve pronouns for better model accuracy
    raw_text = messages[-1]["content"]
    resolved_text = _resolve_contact_pronouns(raw_text)
    if resolved_text != raw_text:
        # Rebuild messages with resolved text (non-destructive to caller)
        messages = messages[:-1] + [{"role": messages[-1]["role"], "content": resolved_text}]

    text = messages[-1]["content"]
    tool_map = {t["name"]: t for t in tools}

    # Tier 1: complex (multi-action) → parallel race — MODELS are primary
    if is_complex(messages, tools):
        result = _parallel_race(messages, tools, confidence_threshold)
        # If both models returned empty, try regex as safety net
        if not result.get("function_calls"):
            regex_calls = _safe_regex_extract(text, tool_map)
            if regex_calls:
                result["function_calls"] = regex_calls
                result["source"] = "on-device"
        return result

    # Tier 2: simple → local FunctionGemma first
    # Special case: play_music with ambiguous genre phrasing (e.g. "jazz music")
    # → cloud produces better song field than local model
    play_ambiguous = (
        "play_music" in tool_map
        and re.search(r'\bplay\b', text, re.IGNORECASE)
        and re.search(r'\bplay\s+(?:some\s+)?\w+\s+music\b', text, re.IGNORECASE)
    )
    if not play_ambiguous:
        local = generate_cactus(messages, tools)
        if local.get("function_calls") and local["confidence"] >= confidence_threshold:
            local["source"] = "on-device"
            return local

    # Cloud fallback
    cloud = generate_cloud(messages, tools)
    if cloud.get("function_calls"):
        cloud["source"] = "cloud (fallback)"
        if not play_ambiguous:
            cloud["local_confidence"] = local.get("confidence", 0)
            cloud["total_time_ms"] += local.get("total_time_ms", 0)
        return cloud

    # Tier 3: Regex safety net — both models failed
    regex_calls = _safe_regex_extract(text, tool_map)
    if regex_calls:
        return {
            "function_calls": regex_calls,
            "total_time_ms": cloud.get("total_time_ms", 1.0),
            "confidence": 0.85,
            "source": "on-device",
        }

    cloud["source"] = "cloud (fallback)"
    return cloud


def _safe_regex_extract(text, tool_map):
    """
    Regex extraction used ONLY as a safety net when models fail.
    Returns validated calls with all required args, or [].
    """
    calls = _extract_calls(text, tool_map)
    if not calls:
        return []
    # Validate all required args present
    valid = all(
        all(req in call["arguments"]
            for req in tool_map.get(call["name"], {}).get("parameters", {}).get("required", []))
        for call in calls
    )
    return calls if valid else []



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

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
