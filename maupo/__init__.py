"""Maupo — a local digital lifeform running on Ollama.

Package layout (extracted from the original single-file qwen_chat.py, one
module per concern; qwen_chat.py stays the entry point):

- maupo.net       Ollama client, one-shot calls, web lookup
- maupo.triggers  phrase detection for memory, search, and face requests
- maupo.notices   the one doorway for background-thread terminal notices
- maupo.memory    sessions, compressor, soft/hard memory, deep recall
- maupo.mind      emotional matrix, persona/speech loading, system prompt
- maupo.engine    realisations, vitality, wondering, heartbeat, face gestures
- maupo.healthlog shared self-diagnosis log (memory/health.log)
"""
