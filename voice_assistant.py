"""
Culinary Knowledge Agent
========================
Ask any cooking question → generate_hybrid routes it
→ FunctionGemma or Gemini selects the right culinary tool
→ Simulated response displayed.

Works on Windows (text mode) with just a GEMINI_API_KEY.
Works on Mac with Cactus + Whisper for full voice mode.

Usage (text, any OS):
    python voice_assistant.py --text

Usage (voice, Mac + Cactus only):
    cactus download whisper-small
    pip install sounddevice scipy
    python voice_assistant.py
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json, os, time, wave, tempfile, re

# Cactus is Mac-only; gracefully degrade on Windows
try:
    from cactus import cactus_init, cactus_transcribe, cactus_destroy
    CACTUS_AVAILABLE = True
except ImportError:
    CACTUS_AVAILABLE = False

from main import generate_hybrid

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


############## Culinary Tool Definitions ##############

TOOLS = [
    {
        "name": "search_recipes",
        "description": "Search for recipes by dish name, ingredient, or cuisine type",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Dish name, ingredient, or cuisine (e.g. 'pasta carbonara', 'chicken', 'Italian')",
                },
                "dietary": {
                    "type": "string",
                    "description": "Optional dietary filter: vegetarian, vegan, gluten-free, dairy-free",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_recipe_details",
        "description": "Get full recipe including ingredients list and step-by-step instructions",
        "parameters": {
            "type": "object",
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Exact or approximate recipe name",
                },
            },
            "required": ["recipe_name"],
        },
    },
    {
        "name": "find_ingredient_substitute",
        "description": "Find a suitable substitute for a cooking ingredient",
        "parameters": {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "Ingredient to substitute (e.g. 'butter', 'eggs', 'buttermilk')",
                },
                "context": {
                    "type": "string",
                    "description": "Optional context for the substitution (e.g. 'baking', 'sauce', 'vegan')",
                },
            },
            "required": ["ingredient"],
        },
    },
    {
        "name": "convert_measurement",
        "description": "Convert cooking measurements between units (cups, grams, ml, tablespoons, etc.)",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. 'cups', 'oz', 'grams')"},
                "to_unit": {"type": "string", "description": "Target unit"},
                "ingredient": {
                    "type": "string",
                    "description": "Optional ingredient for density-based conversions (e.g. 'flour', 'sugar')",
                },
            },
            "required": ["amount", "from_unit", "to_unit"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer (useful for tracking cooking times)",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Number of minutes"},
            },
            "required": ["minutes"],
        },
    },
    {
        "name": "get_nutrition_info",
        "description": "Get nutritional information (calories, macros) for a food or ingredient",
        "parameters": {
            "type": "object",
            "properties": {
                "food": {
                    "type": "string",
                    "description": "Food or ingredient name (e.g. 'avocado', '100g chicken breast')",
                },
            },
            "required": ["food"],
        },
    },
    {
        "name": "find_wine_pairing",
        "description": "Suggest wine pairings for a dish or cuisine",
        "parameters": {
            "type": "object",
            "properties": {
                "dish": {
                    "type": "string",
                    "description": "Dish or cuisine to pair wine with (e.g. 'beef steak', 'seafood pasta')",
                },
            },
            "required": ["dish"],
        },
    },
]


############## Simulated Culinary Tool Responses ##############

# Rich simulated data — replace with real API calls (Spoonacular, Edamam, etc.)

_RECIPES = {
    "pasta carbonara": {
        "ingredients": ["400g spaghetti", "200g pancetta or guanciale", "4 egg yolks",
                        "100g Pecorino Romano", "Black pepper", "Salt"],
        "steps": [
            "Boil salted water and cook spaghetti until al dente.",
            "Fry pancetta in a dry pan until crispy. Remove from heat.",
            "Whisk egg yolks with grated Pecorino and plenty of black pepper.",
            "Reserve 1 cup pasta water. Drain pasta and add to pancetta pan.",
            "Off heat, pour egg mixture over pasta. Toss rapidly, adding pasta water to loosen.",
            "Serve immediately topped with extra Pecorino and black pepper.",
        ],
        "time": "25 min", "servings": 4,
    },
    "chocolate chip cookies": {
        "ingredients": ["2¼ cups flour", "1 tsp baking soda", "1 tsp salt",
                        "1 cup butter (softened)", "¾ cup sugar", "¾ cup brown sugar",
                        "2 eggs", "2 tsp vanilla", "2 cups chocolate chips"],
        "steps": [
            "Preheat oven to 375°F (190°C).",
            "Cream butter and both sugars until fluffy.",
            "Beat in eggs and vanilla.",
            "Mix in flour, baking soda, and salt.",
            "Stir in chocolate chips.",
            "Drop rounded tablespoons onto ungreased baking sheets.",
            "Bake 9–11 minutes until golden brown.",
        ],
        "time": "30 min", "servings": 60,
    },
    "guacamole": {
        "ingredients": ["3 ripe avocados", "1 lime (juiced)", "1 tsp salt",
                        "½ cup diced onion", "3 tbsp cilantro", "2 roma tomatoes (diced)",
                        "1 tsp garlic (minced)", "1 jalapeño (minced)"],
        "steps": [
            "Halve and pit avocados. Scoop flesh into a bowl.",
            "Mash to your desired consistency.",
            "Add lime juice and salt. Mix well.",
            "Fold in onion, cilantro, tomatoes, garlic, and jalapeño.",
            "Taste and adjust seasoning. Serve immediately.",
        ],
        "time": "10 min", "servings": 6,
    },
}

_SUBSTITUTES = {
    "butter": "Use equal amount of coconut oil, or ¾ amount of vegetable oil. For baking, applesauce works at ½ ratio.",
    "eggs": "Per egg: 1 tbsp ground flaxseed + 3 tbsp water (let sit 5 min), or ¼ cup unsweetened applesauce, or 3 tbsp aquafaba.",
    "buttermilk": "Mix 1 cup milk + 1 tbsp white vinegar or lemon juice. Let sit 5 min until curdled.",
    "heavy cream": "Mix ⅔ cup whole milk + ⅓ cup melted butter. For whipping cream, use full-fat coconut cream.",
    "sugar": "Use ¾ cup honey or maple syrup per 1 cup sugar (reduce other liquids by ¼). Or use coconut sugar 1:1.",
    "sour cream": "Plain Greek yogurt works 1:1 in most recipes.",
    "flour": "For gluten-free: use almond flour (1:1 for cookies/cakes) or a GF blend. For thickening: cornstarch at half the amount.",
    "milk": "Use plant-based milk (oat, almond, soy) 1:1 in most recipes.",
    "baking powder": "Use ¼ tsp baking soda + ½ tsp cream of tartar per 1 tsp baking powder.",
    "vanilla extract": "Use 1 tsp vanilla bean paste, or ¼ tsp vanilla powder per 1 tsp extract.",
}

_NUTRITION = {
    "avocado": "Per 100g: 160 kcal | Fat: 15g | Carbs: 9g | Fiber: 7g | Protein: 2g | Potassium: 485mg",
    "chicken breast": "Per 100g (cooked): 165 kcal | Fat: 3.6g | Carbs: 0g | Protein: 31g",
    "egg": "Per large egg: 72 kcal | Fat: 5g | Carbs: 0.4g | Protein: 6g",
    "pasta": "Per 100g (dry): 371 kcal | Fat: 1.5g | Carbs: 75g | Fiber: 3g | Protein: 13g",
    "butter": "Per 100g: 717 kcal | Fat: 81g (Saturated: 51g) | Sodium: 11mg | Protein: 0.9g",
    "broccoli": "Per 100g: 34 kcal | Fat: 0.4g | Carbs: 7g | Fiber: 2.6g | Protein: 2.8g | Vitamin C: 89mg",
    "salmon": "Per 100g: 208 kcal | Fat: 13g | Carbs: 0g | Protein: 20g | Omega-3: 2.3g",
}

_WINE_PAIRINGS = {
    "beef": "Full-bodied reds: Cabernet Sauvignon, Malbec, or Syrah. Bold tannins complement the rich meat.",
    "pasta carbonara": "Crisp whites: Pinot Grigio or Vermentino. Their acidity cuts through the creamy sauce. Or a light Chianti.",
    "seafood": "Crisp whites: Sauvignon Blanc, Albariño, or Chablis. Avoid heavy reds.",
    "chicken": "Versatile — Chardonnay for creamy dishes, Pinot Noir for roasted, Rosé for grilled.",
    "lamb": "Bordeaux-style blends or Syrah. Rhône reds work beautifully.",
    "pizza": "Chianti, Barbera d'Asti, or a light Zinfandel. Something casual and medium-bodied.",
    "chocolate": "Port (ruby or tawny), Banyuls, or a dark-fruit Zinfandel for dessert pairing.",
    "spicy": "Off-dry Riesling or Gewürztraminer — the sweetness tempers heat. Avoid high-tannin reds.",
}

_CONVERSIONS = {
    # (from_unit, to_unit): (factor_or_func, note)
    ("cups", "ml"):     (240, "1 cup = 240ml"),
    ("ml", "cups"):     (1/240, "240ml = 1 cup"),
    ("cups", "grams"):  (None, "ingredient-specific — rough: flour≈120g, sugar≈200g, butter≈225g"),
    ("oz", "grams"):    (28.35, "1 oz = 28.35g"),
    ("grams", "oz"):    (1/28.35, "28.35g = 1 oz"),
    ("tbsp", "ml"):     (15, "1 tbsp = 15ml"),
    ("ml", "tbsp"):     (1/15, "1 tbsp = 15ml"),
    ("tsp", "ml"):      (5, "1 tsp = 5ml"),
    ("ml", "tsp"):      (1/5, "1 tsp = 5ml"),
    ("cups", "tbsp"):   (16, "1 cup = 16 tbsp"),
    ("tbsp", "cups"):   (1/16, "16 tbsp = 1 cup"),
    ("lbs", "grams"):   (453.6, "1 lb = 453.6g"),
    ("grams", "lbs"):   (1/453.6, "453.6g = 1 lb"),
    ("f", "c"):         ("fahrenheit_to_celsius", "°F to °C"),
    ("c", "f"):         ("celsius_to_fahrenheit", "°C to °F"),
}


def execute_tool(name, arguments):
    """Simulate culinary tool execution with rich, helpful output."""

    if name == "search_recipes":
        q = arguments.get("query", "").lower()
        dietary = arguments.get("dietary", "")
        # Find partial matches
        matches = [r for r in _RECIPES if q in r]
        if matches:
            results = ", ".join(m.title() for m in matches[:3])
            dietary_note = f" (filtered: {dietary})" if dietary else ""
            return (f"Found recipes for '{q}'{dietary_note}:\n"
                    f"  → {results}\n"
                    f"  Ask for details on any of these to get ingredients & steps.")
        # Generic response
        cuisines = {"italian": "Pasta Carbonara, Risotto, Osso Buco",
                    "indian": "Butter Chicken, Dal Makhani, Biryani",
                    "mexican": "Tacos al Pastor, Enchiladas, Guacamole",
                    "chinese": "Kung Pao Chicken, Mapo Tofu, Fried Rice",
                    "french": "Coq au Vin, Ratatouille, Crème Brûlée"}
        for cuisine, dishes in cuisines.items():
            if cuisine in q:
                return f"Popular {cuisine.title()} recipes: {dishes}."
        return (f"Recipes matching '{q}':\n"
                f"  Try these classics: Pasta Carbonara, Chicken Curry, Beef Stir-fry.\n"
                f"  Ask me for details on any dish!")

    if name == "get_recipe_details":
        recipe = arguments.get("recipe_name", "").lower()
        # Find best match
        match = None
        for key in _RECIPES:
            if key in recipe or recipe in key:
                match = key
                break
        if match:
            r = _RECIPES[match]
            ing = "\n  ".join(r["ingredients"])
            steps = "\n  ".join(f"{i+1}. {s}" for i, s in enumerate(r["steps"]))
            return (f"📋 {match.title()}  ({r['time']} | Serves {r['servings']})\n\n"
                    f"Ingredients:\n  {ing}\n\n"
                    f"Steps:\n  {steps}")
        return (f"I don't have a built-in recipe for '{recipe}', but here's a starting point:\n"
                f"Search 'seriouseats.com' or 'allrecipes.com' for tested recipes.\n"
                f"I can help with substitutions, conversions, and timers once you start cooking!")

    if name == "find_ingredient_substitute":
        ing = arguments.get("ingredient", "").lower().strip()
        ctx = arguments.get("context", "")
        # Find match
        for key, sub in _SUBSTITUTES.items():
            if key in ing or ing in key:
                ctx_note = f" (for {ctx})" if ctx else ""
                return f"Substitute for {ing}{ctx_note}:\n  {sub}"
        return (f"For '{ing}', consider:\n"
                f"  • Check if the recipe can work without it entirely\n"
                f"  • Search 'substitute for {ing} in cooking' — Serious Eats has great guides\n"
                f"  • For baking, changes affect texture/rise — be cautious")

    if name == "convert_measurement":
        amount = arguments.get("amount", 1)
        from_u = arguments.get("from_unit", "").lower().strip()
        to_u = arguments.get("to_unit", "").lower().strip()
        ingredient = arguments.get("ingredient", "")

        key = (from_u, to_u)
        if key in _CONVERSIONS:
            factor, note = _CONVERSIONS[key]
            if factor == "fahrenheit_to_celsius":
                result = (amount - 32) * 5 / 9
                return f"{amount}°F = {result:.1f}°C  ({note})"
            elif factor == "celsius_to_fahrenheit":
                result = amount * 9 / 5 + 32
                return f"{amount}°C = {result:.1f}°F  ({note})"
            elif factor is None:
                # Ingredient-based volume-to-mass
                density = {"flour": 120, "sugar": 200, "butter": 225,
                           "rice": 185, "oats": 90, "cocoa": 85}
                ing_lower = ingredient.lower() if ingredient else ""
                for k, g_per_cup in density.items():
                    if k in ing_lower:
                        cups = amount  # assuming from_unit=cups
                        grams = cups * g_per_cup
                        return f"{amount} cup{'s' if amount != 1 else ''} {k} ≈ {grams:.0f}g  (density estimate)"
                return f"{amount} {from_u} → {to_u}: depends on ingredient. {note}"
            else:
                result = amount * factor
                return f"{amount} {from_u} = {result:.2f} {to_u}  ({note})"
        # Reverse lookup hint
        reverse = (to_u, from_u)
        if reverse in _CONVERSIONS:
            f2, note = _CONVERSIONS[reverse]
            if isinstance(f2, (int, float)):
                result = amount / f2
                return f"{amount} {from_u} = {result:.2f} {to_u}  ({note})"
        return f"Conversion from {from_u} to {to_u}: not found. Try specifying units like 'cups', 'grams', 'ml', 'tbsp', 'oz'."

    if name == "set_timer":
        mins = arguments.get("minutes", 0)
        label = f"{mins} minute{'s' if mins != 1 else ''}"
        tips = {
            1: "1 min — perfect for toasting nuts or melting butter.",
            3: "3 min — soft-boiled egg time!",
            5: "5 min — rest a steak this long after cooking.",
            10: "10 min — resting time for a roast.",
            12: "12 min — al dente pasta (check your package).",
            15: "15 min — bread rolls are likely done. Check with a thermometer!",
            20: "20 min — most rice is cooked by now.",
            25: "25 min — check on those roasted vegetables.",
            30: "30 min — cookies or small cakes may be ready.",
            45: "45 min — whole chicken pieces should be cooked through.",
            60: "1 hour — whole roast chicken is likely done (check 165°F / 74°C inside).",
        }
        tip = tips.get(mins, "")
        tip_str = f"\n  💡 {tip}" if tip else ""
        return f"⏱  Timer set for {label}. Starting now!{tip_str}"

    if name == "get_nutrition_info":
        food = arguments.get("food", "").lower().strip()
        for key, info in _NUTRITION.items():
            if key in food or food in key:
                return f"🥗 Nutrition — {food.title()}:\n  {info}"
        return (f"Nutrition for '{food}':\n"
                f"  I don't have that in my local database.\n"
                f"  Try: cronometer.com or ndb.nal.usda.gov for verified data.")

    if name == "find_wine_pairing":
        dish = arguments.get("dish", "").lower().strip()
        for key, pairing in _WINE_PAIRINGS.items():
            if key in dish or dish in key:
                return f"🍷 Wine pairing for {dish}:\n  {pairing}"
        # Generic rules
        if any(w in dish for w in ["fish", "shrimp", "lobster", "crab", "scallop"]):
            return f"🍷 Wine pairing for {dish}:\n  {_WINE_PAIRINGS['seafood']}"
        if "beef" in dish or "steak" in dish or "burger" in dish:
            return f"🍷 Wine pairing for {dish}:\n  {_WINE_PAIRINGS['beef']}"
        return (f"🍷 Wine pairing for {dish}:\n"
                f"  General rule: red meat → bold reds (Cab Sauv, Malbec). "
                f"White meat/fish → crisp whites (Sauvignon Blanc, Pinot Grigio). "
                f"Spicy food → off-dry Riesling.")

    return f"Executed {name} with {arguments}"


############## Audio recording ##############

def record_audio(duration=5, sample_rate=16000):
    """Record from mic for `duration` seconds, return as numpy array."""
    print(f"  Recording for {duration}s... speak now!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="int16")
    sd.wait()
    print("  Recording complete.")
    return audio, sample_rate


def save_wav(audio, sample_rate):
    """Save numpy audio array to a temp WAV file, return path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp.name


############## Main agent loop ##############

def run(whisper_path="cactus/weights/whisper-small", use_voice=True):
    print("\n" + "=" * 62)
    print("  🍽  Culinary Knowledge Agent")
    print("  Powered by FunctionGemma + Gemini hybrid routing")
    print("=" * 62)
    print("\nExample queries:")
    print("  • How do I make pasta carbonara?")
    print("  • What can I substitute for eggs in baking?")
    print("  • Convert 2 cups of flour to grams")
    print("  • Set a timer for 12 minutes")
    print("  • What wine goes with beef steak?")
    print("  • Calories in an avocado?")
    print("  • Find me a vegetarian Italian recipe")

    whisper = None
    if use_voice and AUDIO_AVAILABLE and CACTUS_AVAILABLE:
        print("\nLoading Whisper (on-device transcription)...")
        whisper = cactus_init(whisper_path)
        print("Whisper ready.\n")
    elif use_voice:
        print("\n[info] Voice mode unavailable (needs Cactus + sounddevice on Mac). Falling back to text.\n")

    WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

    try:
        while True:
            print("\n" + "-" * 62)

            if whisper and AUDIO_AVAILABLE:
                input("Press Enter and speak your culinary question (5 seconds)...")
                audio, sr = record_audio(duration=5)
                wav_path = save_wav(audio, sr)

                t0 = time.time()
                raw = cactus_transcribe(whisper, wav_path, prompt=WHISPER_PROMPT)
                transcribe_ms = (time.time() - t0) * 1000
                os.unlink(wav_path)

                transcript = json.loads(raw).get("response", "").strip()
                if not transcript:
                    print("  [could not transcribe, try again]")
                    continue

                print(f"\n  Heard: \"{transcript}\"")
                print(f"  Transcription: {transcribe_ms:.0f}ms (on-device Whisper)")
            else:
                transcript = input("\nAsk a culinary question (or 'quit'): ").strip()
                if transcript.lower() in ("quit", "exit", "q"):
                    break
                if not transcript:
                    continue
                transcribe_ms = 0

            # Route through hybrid system
            messages = [{"role": "user", "content": transcript}]
            t1 = time.time()
            result = generate_hybrid(messages, TOOLS)
            route_ms = (time.time() - t1) * 1000

            source = result.get("source", "unknown")
            model_ms = result.get("total_time_ms", route_ms)
            calls = result.get("function_calls", [])

            print(f"\n  Routing: {source}  |  model: {model_ms:.0f}ms", end="")
            if transcribe_ms:
                total = transcribe_ms + model_ms
                print(f"  |  total (incl. transcription): {total:.0f}ms", end="")
            print()

            if not calls:
                print("\n  [no culinary tool matched — try rephrasing]")
                print("  Hint: ask about recipes, substitutes, conversions, timers, nutrition, or wine pairings.")
                continue

            print()
            for call in calls:
                output = execute_tool(call["name"], call["arguments"])
                print(f"  [{call['name']}]\n{output}\n")

    except KeyboardInterrupt:
        print("\n\nBon appétit! 👨‍🍳")
    finally:
        if whisper and CACTUS_AVAILABLE:
            cactus_destroy(whisper)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Culinary Knowledge Agent")
    parser.add_argument("--text", action="store_true", help="Use text input (works on Windows, no mic needed)")
    parser.add_argument("--whisper", default="cactus/weights/whisper-small", help="Whisper model path (Mac only)")
    args = parser.parse_args()

    run(whisper_path=args.whisper, use_voice=not args.text)
