"""
benchmark_unseen.py

Tests generate_hybrid on tools that are NOT hardcoded in main.py's _extract_calls.
All cases here will fall through fast_path and require on-device or cloud inference,
directly measuring the model's ability to generalise to unseen tools.
"""

import sys, os
sys.path.insert(0, "cactus/python/src")
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

from main import generate_hybrid
from benchmark import compute_f1, run_benchmark

############## New tool definitions (NOT hardcoded in main.py _extract_calls) ##############

TOOL_TRANSLATE_TEXT = {
    "name": "translate_text",
    "description": "Translate text into a target language",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to translate"},
            "language": {"type": "string", "description": "Target language (e.g. Spanish, French)"},
        },
        "required": ["text", "language"],
    },
}

TOOL_BOOK_RIDE = {
    "name": "book_ride",
    "description": "Book a ride to a destination",
    "parameters": {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "Destination address or place name"},
        },
        "required": ["destination"],
    },
}

TOOL_SEND_EMAIL = {
    "name": "send_email",
    "description": "Send an email to a recipient",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email address"},
            "body": {"type": "string", "description": "Body of the email"},
        },
        "required": ["to", "body"],
    },
}

TOOL_CONTROL_LIGHTS = {
    "name": "control_lights",
    "description": "Control smart home lights in a room",
    "parameters": {
        "type": "object",
        "properties": {
            "room": {"type": "string", "description": "Room name (e.g. living room, bedroom)"},
            "action": {"type": "string", "description": "Action to take: on or off"},
        },
        "required": ["room", "action"],
    },
}

TOOL_GET_NEWS = {
    "name": "get_news",
    "description": "Fetch the latest news headlines for a topic",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "News topic (e.g. technology, sports)"},
        },
        "required": ["topic"],
    },
}

TOOL_ADD_TO_CART = {
    "name": "add_to_cart",
    "description": "Add an item to the shopping cart",
    "parameters": {
        "type": "object",
        "properties": {
            "item": {"type": "string", "description": "Item name to add"},
            "quantity": {"type": "integer", "description": "Quantity to add"},
        },
        "required": ["item", "quantity"],
    },
}

TOOL_OPEN_APP = {
    "name": "open_app",
    "description": "Open an application by name",
    "parameters": {
        "type": "object",
        "properties": {
            "app": {"type": "string", "description": "Application name"},
        },
        "required": ["app"],
    },
}

TOOL_CHECK_CALENDAR = {
    "name": "check_calendar",
    "description": "Check calendar events for a given date",
    "parameters": {
        "type": "object",
        "properties": {
            "date": {"type": "string", "description": "Date to check (e.g. today, tomorrow, 2025-06-01)"},
        },
        "required": ["date"],
    },
}

############## Benchmark cases ##############

BENCHMARKS = [
    # ===== Easy: 1 new tool, direct request =====
    {
        "name": "translate_hello",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Translate 'hello world' to Spanish."}],
        "tools": [TOOL_TRANSLATE_TEXT],
        "expected_calls": [{"name": "translate_text", "arguments": {"text": "hello world", "language": "Spanish"}}],
    },
    {
        "name": "book_ride_airport",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Book a ride to the airport."}],
        "tools": [TOOL_BOOK_RIDE],
        "expected_calls": [{"name": "book_ride", "arguments": {"destination": "airport"}}],
    },
    {
        "name": "send_email_basic",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Send an email to john@example.com saying I'll be there at 5."}],
        "tools": [TOOL_SEND_EMAIL],
        "expected_calls": [{"name": "send_email", "arguments": {"to": "john@example.com", "body": "I'll be there at 5"}}],
    },
    {
        "name": "lights_off",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Turn off the living room lights."}],
        "tools": [TOOL_CONTROL_LIGHTS],
        "expected_calls": [{"name": "control_lights", "arguments": {"room": "living room", "action": "off"}}],
    },
    {
        "name": "open_spotify",
        "difficulty": "easy",
        "messages": [{"role": "user", "content": "Open Spotify."}],
        "tools": [TOOL_OPEN_APP],
        "expected_calls": [{"name": "open_app", "arguments": {"app": "Spotify"}}],
    },

    # ===== Medium: pick the right new tool from several =====
    {
        "name": "get_tech_news",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Get me the latest tech news."}],
        "tools": [TOOL_GET_NEWS, TOOL_SEND_EMAIL, TOOL_BOOK_RIDE],
        "expected_calls": [{"name": "get_news", "arguments": {"topic": "tech"}}],
    },
    {
        "name": "add_milk_to_cart",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Add 2 milks to my cart."}],
        "tools": [TOOL_ADD_TO_CART, TOOL_OPEN_APP, TOOL_CHECK_CALENDAR],
        "expected_calls": [{"name": "add_to_cart", "arguments": {"item": "milk", "quantity": 2}}],
    },
    {
        "name": "check_tomorrow_calendar",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "What's on my calendar tomorrow?"}],
        "tools": [TOOL_CHECK_CALENDAR, TOOL_SEND_EMAIL, TOOL_CONTROL_LIGHTS],
        "expected_calls": [{"name": "check_calendar", "arguments": {"date": "tomorrow"}}],
    },
    {
        "name": "translate_among_new",
        "difficulty": "medium",
        "messages": [{"role": "user", "content": "Translate 'good night' to French."}],
        "tools": [TOOL_TRANSLATE_TEXT, TOOL_SEND_EMAIL, TOOL_CONTROL_LIGHTS, TOOL_OPEN_APP],
        "expected_calls": [{"name": "translate_text", "arguments": {"text": "good night", "language": "French"}}],
    },

    # ===== Hard: multiple new tools in one request =====
    {
        "name": "translate_and_email",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Translate 'see you tomorrow' to French and send it to alice@mail.com."}],
        "tools": [TOOL_TRANSLATE_TEXT, TOOL_SEND_EMAIL, TOOL_BOOK_RIDE],
        "expected_calls": [
            {"name": "translate_text", "arguments": {"text": "see you tomorrow", "language": "French"}},
            {"name": "send_email", "arguments": {"to": "alice@mail.com", "body": "see you tomorrow"}},
        ],
    },
    {
        "name": "ride_and_lights",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Book a ride to downtown and turn off the bedroom lights."}],
        "tools": [TOOL_BOOK_RIDE, TOOL_CONTROL_LIGHTS, TOOL_GET_NEWS, TOOL_OPEN_APP],
        "expected_calls": [
            {"name": "book_ride", "arguments": {"destination": "downtown"}},
            {"name": "control_lights", "arguments": {"room": "bedroom", "action": "off"}},
        ],
    },
    {
        "name": "cart_and_calendar",
        "difficulty": "hard",
        "messages": [{"role": "user", "content": "Add 3 apples to my cart and check my calendar for today."}],
        "tools": [TOOL_ADD_TO_CART, TOOL_CHECK_CALENDAR, TOOL_SEND_EMAIL, TOOL_TRANSLATE_TEXT],
        "expected_calls": [
            {"name": "add_to_cart", "arguments": {"item": "apples", "quantity": 3}},
            {"name": "check_calendar", "arguments": {"date": "today"}},
        ],
    },
]

if __name__ == "__main__":
    run_benchmark(BENCHMARKS)
