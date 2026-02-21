
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

def fast_path(messages, tools):
    """
    Rule-based instant handler for simple single-tool requests.
    Returns a result dict immediately (no model inference) or None if no pattern matched.
    Filled in by Windows User 1 — stub returns None until then.
    """
    return None  # TODO: Windows User 1 fills this in


def _on_device_result(calls, time_ms=0.5):
    """Build an on-device result dict in the standard format."""
    return {
        "function_calls": calls,
        "total_time_ms": time_ms,
        "confidence": 1.0,
        "source": "on-device",
    }


def _count_implied_actions(text):
    """
    Count how many distinct actions are implied in the user message.
    Looks for conjunctions that chain separate requests.
    e.g. "Set an alarm AND check the weather AND play music" → 3
    """
    # Each conjunction likely introduces another action
    conjunctions = re.findall(
        r'\band\b|\balso\b|\bplus\b|\bthen\b|\bas well\b',
        text.lower()
    )
    # Count verbs that typically start a new request
    action_verbs = re.findall(
        r'\b(?:set|send|text|play|check|get|find|search|remind|create|look up|wake)\b',
        text.lower()
    )
    return max(len(conjunctions) + 1, len(action_verbs))


def is_complex(messages, tools):
    """
    Classify whether a request is complex (multi-action / hard).
    Complex = needs multiple tool calls, so local model often fails.
    Triggers the parallel race strategy instead of local-only.
    """
    text = messages[-1]["content"].lower()
    implied = _count_implied_actions(text)

    # More than one action implied → complex
    if implied >= 2:
        return True

    # Many tools available increases ambiguity → be cautious
    if len(tools) >= 5:
        return True

    return False


def _parallel_race(messages, tools, confidence_threshold):
    """
    Fire local (Cactus) and cloud (Gemini) simultaneously.
    - If local finishes with sufficient confidence → use it (saves latency + scores on-device)
    - Otherwise → use cloud result (accuracy wins)
    Wall time = max(local_time, cloud_time) rather than sum.
    """
    wall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        local_future = executor.submit(generate_cactus, messages, tools)
        cloud_future = executor.submit(generate_cloud, messages, tools)

        local = local_future.result()
        cloud = cloud_future.result()

    wall_time_ms = (time.time() - wall_start) * 1000

    if local["confidence"] >= confidence_threshold and local.get("function_calls"):
        local["source"] = "on-device"
        local["total_time_ms"] = wall_time_ms
        return local

    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local["confidence"]
    cloud["total_time_ms"] = wall_time_ms
    return cloud


def generate_hybrid(messages, tools, confidence_threshold=0.75):
    """
    Smart hybrid routing strategy:

    1. Fast path  — rule-based instant answer for simple, predictable requests.
                    Zero model inference, tagged on-device. (<1ms)
    2. Complex    — parallel race: fire local + cloud simultaneously.
                    Use local if confidence is high enough, else cloud.
                    Wall time = max(local, cloud) instead of sum.
    3. Simple     — local first with tuned confidence threshold.
                    Fall back to cloud only when local is uncertain.
    """
    # Step 1: Rule-based fast path (Windows User 1)
    result = fast_path(messages, tools)
    if result is not None:
        return result

    # Step 2: Parallel race for complex / multi-action requests
    if is_complex(messages, tools):
        return _parallel_race(messages, tools, confidence_threshold)

    # Step 3: Local-first for simple single-tool requests
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

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
