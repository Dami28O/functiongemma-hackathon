"""
Standalone Culinary Knowledge Agent
===================================
Ask any cooking question → Direct Gemini API (via requests) 
→ Simulated response displayed.

DESIGN:
- No SDK dependencies: Uses 'requests' for direct Gemini API calls.
- Standalone on Windows: Works without main.py or cactus.
- Full Voice Mode on Mac: Uses Cactus/Whisper if available.

Usage:
    python voice_assistant.py --text
"""

import sys
import json
import os
import time
import re
import requests

# Load .env manually to avoid extra dependencies
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

# Model and API Configuration
MODEL_NAME = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# Optional Dependencies for Mac/Audio
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    sys.path.insert(0, "cactus/python/src")
    from cactus import cactus_init, cactus_transcribe, cactus_destroy
    CACTUS_AVAILABLE = True
except ImportError:
    CACTUS_AVAILABLE = False


############## Culinary Tool Definitions ##############

TOOLS = [
    {
        "name": "search_recipes",
        "description": "Search for recipes by dish name, ingredient, or cuisine type",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Dish, ingredient, or cuisine"},
                "dietary": {"type": "string", "description": "vegetarian, vegan, etc."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recipe_details",
        "description": "Get ingredients and instructions for a specific recipe",
        "parameters": {
            "type": "object",
            "properties": {"recipe_name": {"type": "string"}},
            "required": ["recipe_name"],
        },
    },
    {
        "name": "find_ingredient_substitute",
        "description": "Find a substitute for a cooking ingredient",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {"type": "string"},
                "context": {"type": "string"},
            },
            "required": ["ingredient"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer for cooking",
        "parameters": {
            "type": "object",
            "properties": {"minutes": {"type": "integer"}},
            "required": ["minutes"],
        },
    },
    {
        "name": "get_nutrition_info",
        "description": "Get calories and macros for a food",
        "parameters": {
            "type": "object",
            "properties": {"food": {"type": "string"}},
            "required": ["food"],
        },
    },
    {
        "name": "find_wine_pairing",
        "description": "Suggest wine pairings for a dish",
        "parameters": {
            "type": "object",
            "properties": {"dish": {"type": "string"}},
            "required": ["dish"],
        },
    },
]


############## Direct API Generation Logic ##############

def generate_culinary(text):
    """
    Directly calls Gemini API via requests (bypass library issues).
    """
    # 1. Check for Timer (Regex fast-path)
    timer_match = re.search(r'(?:set|start|give me)\s+(?:a\s+)?(?:timer|countdown)\s+(?:for\s+)?(\d+)\s+min', text, re.I)
    if timer_match:
        return {
            "function_calls": [{"name": "set_timer", "arguments": {"minutes": int(timer_match.group(1))}}],
            "source": "on-device (fast-path)",
            "total_time_ms": 1.0
        }

    # 2. Call Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"error": "Missing GEMINI_API_KEY", "function_calls": []}

    t0 = time.time()
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "tools": [{"function_declarations": TOOLS}]
    }
    
    try:
        res = requests.post(f"{API_URL}?key={api_key}", json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        calls = []
        if "candidates" in data:
            parts = data["candidates"][0]["content"]["parts"]
            for part in parts:
                if "functionCall" in part:
                    fn = part["functionCall"]
                    calls.append({
                        "name": fn["name"],
                        "arguments": fn["args"]
                    })
        
        return {
            "function_calls": calls,
            "source": "cloud (Gemini)",
            "total_time_ms": (time.time() - t0) * 1000
        }
    except Exception as e:
        return {"error": str(e), "function_calls": []}


############## Culinary Simulations ##############

_RECIPES = {
    "pasta carbonara": "Ingredients: Spaghetti, Eggs, Guanciale, Pecorino, Pepper.\nSteps: 1. Boil pasta. 2. Fry pork. 3. Mix eggs/cheese. 4. Combine off-heat.",
    "chocolate chip cookies": "Ingredients: Flour, Butter, Sugar, Eggs, Choc Chips.\nSteps: 1. Cream butter/sugar. 2. Add egg. 3. Mix dry. 4. Bake 10m at 375F."
}

def execute_tool(name, args):
    if name == "search_recipes":
        return f"Found matching recipes for '{args.get('query')}': Pasta Carbonara, Tiramisu, Risotto."
    if name == "get_recipe_details":
        rn = args.get("recipe_name", "").lower()
        if "pasta" in rn or "carbonara" in rn: return _RECIPES["pasta carbonara"]
        return f"Recipe details for {rn}: Standard instructions found in Culinary DB."
    if name == "find_ingredient_substitute":
        return f"For {args.get('ingredient')}, try using {args.get('context', 'a similar alternative')} like honey instead of sugar."
    if name == "set_timer":
        return f"⏱ Timer set for {args.get('minutes')} minutes. Good luck with the cooking!"
    if name == "get_nutrition_info":
        return f"Nutrition info for {args.get('food')}: 150 calories, 5g protein, 10g fat."
    if name == "find_wine_pairing":
        return f"Wine pairing for {args.get('dish')}: A dry Pinot Grigio would complement this perfectly."
    return f"Action {name} executed."


############## Agent Loop ##############

def run():
    print("\n" + "👨‍🍳" * 20)
    print("  CULINARY EXPERT AGENT")
    print("  (Standalone Mode)")
    print("👨‍🍳" * 20 + "\n")

    while True:
        try:
            transcript = input("\nAsk culinary question (or 'q' to quit): ").strip()
            if transcript.lower() in ('q', 'quit'): break
            if not transcript: continue

            print(f"  Thinking...")
            result = generate_culinary(transcript)

            if "error" in result:
                print(f"  [API Error] {result['error']}")
                continue

            calls = result.get("function_calls", [])
            if not calls:
                print("  [Gemini replied, but no tool was called. Try asking for a recipe!]")
                continue

            for call in calls:
                print(f"\n  [{call['name'].upper()}]")
                print(f"  {execute_tool(call['name'], call['arguments'])}")
            
            print(f"\n  (Route: {result['source']} | time: {result['total_time_ms']:.0f}ms)")

        except KeyboardInterrupt:
            break
    print("\nBon Appétit!")

if __name__ == "__main__":
    run()
