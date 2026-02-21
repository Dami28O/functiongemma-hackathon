"""
Voice-to-Action Assistant
=========================
Speak a command → Whisper transcribes on-device → generate_hybrid routes it
→ FunctionGemma or Gemini executes the tool call → result displayed.

Setup (Mac only — needs Cactus):
    cactus download whisper-small
    pip install sounddevice scipy

Usage:
    python voice_assistant.py
"""

import sys
sys.path.insert(0, "cactus/python/src")

import json, os, time, wave, tempfile
import numpy as np

from cactus import cactus_init, cactus_transcribe, cactus_destroy
from main import generate_hybrid

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[warning] sounddevice not installed — run: pip install sounddevice scipy")
    print("[warning] Falling back to text input mode.\n")


############## Tool definitions (same as benchmark) ##############

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City name"}},
            "required": ["location"],
        },
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a given time",
        "parameters": {
            "type": "object",
            "properties": {
                "hour": {"type": "integer", "description": "Hour (0-23)"},
                "minute": {"type": "integer", "description": "Minute (0-59)"},
            },
            "required": ["hour", "minute"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Recipient name"},
                "message": {"type": "string", "description": "Message content"},
            },
            "required": ["recipient", "message"],
        },
    },
    {
        "name": "create_reminder",
        "description": "Create a reminder with a title and time",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Reminder title"},
                "time": {"type": "string", "description": "Time e.g. 3:00 PM"},
            },
            "required": ["title", "time"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search for a contact by name",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Name to search"}},
            "required": ["query"],
        },
    },
    {
        "name": "play_music",
        "description": "Play a song or playlist",
        "parameters": {
            "type": "object",
            "properties": {"song": {"type": "string", "description": "Song or playlist name"}},
            "required": ["song"],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer",
        "parameters": {
            "type": "object",
            "properties": {"minutes": {"type": "integer", "description": "Minutes"}},
            "required": ["minutes"],
        },
    },
]


############## Simulated tool execution ##############

def execute_tool(name, arguments):
    """Simulate real tool execution with formatted output."""
    if name == "get_weather":
        city = arguments.get("location", "unknown")
        return f"Weather in {city}: 22°C, partly cloudy. Humidity 65%."

    if name == "set_alarm":
        h = arguments.get("hour", 0)
        m = arguments.get("minute", 0)
        period = "AM" if h < 12 else "PM"
        h12 = h % 12 or 12
        return f"Alarm set for {h12}:{m:02d} {period}."

    if name == "send_message":
        r = arguments.get("recipient", "unknown")
        msg = arguments.get("message", "")
        return f"Message sent to {r}: \"{msg}\""

    if name == "create_reminder":
        title = arguments.get("title", "")
        t = arguments.get("time", "")
        return f"Reminder created: \"{title}\" at {t}."

    if name == "search_contacts":
        q = arguments.get("query", "")
        return f"Found contact: {q} — +1 (555) 010-1234"

    if name == "play_music":
        song = arguments.get("song", "")
        return f"Now playing: {song} 🎵"

    if name == "set_timer":
        mins = arguments.get("minutes", 0)
        return f"Timer set for {mins} minute{'s' if mins != 1 else ''}. Starting now."

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
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp.name


############## Main assistant loop ##############

def run(whisper_path="cactus/weights/whisper-small", use_voice=True):
    print("\n" + "=" * 60)
    print("  Voice-to-Action Assistant")
    print("  Powered by FunctionGemma + Gemini hybrid routing")
    print("=" * 60)

    whisper = None
    if use_voice and AUDIO_AVAILABLE:
        print("\nLoading Whisper (on-device transcription)...")
        whisper = cactus_init(whisper_path)
        print("Whisper ready.\n")

    WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

    try:
        while True:
            print("\n" + "-" * 60)

            if whisper and AUDIO_AVAILABLE:
                input("Press Enter and speak your command (5 seconds)...")
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
                transcript = input("Type your command (or 'quit'): ").strip()
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
                print("  [no tool call recognised — try rephrasing]")
                continue

            print()
            for call in calls:
                output = execute_tool(call["name"], call["arguments"])
                print(f"  [{call['name']}] {output}")

    except KeyboardInterrupt:
        print("\n\nBye!")
    finally:
        if whisper:
            cactus_destroy(whisper)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice-to-Action Assistant")
    parser.add_argument("--text", action="store_true", help="Use text input instead of mic")
    parser.add_argument("--whisper", default="cactus/weights/whisper-small", help="Whisper model path")
    args = parser.parse_args()

    run(whisper_path=args.whisper, use_voice=not args.text)
