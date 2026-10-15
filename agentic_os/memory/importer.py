"""Import memory exported from other AI assistants.

Supported inputs (auto-detected):

* **Saved memories as text**: the list an assistant shows under its memory /
  personalization settings (ChatGPT "Manage memories", Claude memory, Gemini
  "Saved info", Grok memories...), pasted or uploaded as .txt/.md. Each line
  or bullet becomes a fact.
* **ChatGPT data export**: ``conversations.json`` (or the whole .zip).
* **Claude data export**: ``conversations.json`` with ``chat_messages``.
* **Google Takeout (Gemini Apps activity)** in JSON format.
* **Any other JSON export** (DeepSeek, Mistral Le Chat, Grok, ...): message
  objects with a role/author/sender and content/text are found heuristically.

Conversations become *episodes* (what you talked about); explicit memories
become *facts*. Optionally a model distills durable facts from conversations.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any

MAX_UNZIPPED = 300 * 1024 * 1024  # zip-bomb guard
USER_ROLES = {"user", "human", "you", "me"}
ASSISTANT_ROLES = {"assistant", "model", "ai", "bot", "gpt", "claude", "gemini", "grok", "chatbot", "system_assistant"}


@dataclass
class Conversation:
    title: str
    messages: list[tuple[str, str]] = field(default_factory=list)  # (role, text)

    def user_text(self, limit: int = 1200) -> str:
        text = " | ".join(t for r, t in self.messages if r == "user")
        return text[:limit]


@dataclass
class ParsedExport:
    source: str
    conversations: list[Conversation] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)


def _norm_role(role: Any) -> str | None:
    if isinstance(role, dict):
        role = role.get("role") or role.get("name")
    role = str(role or "").strip().lower()
    if role in USER_ROLES:
        return "user"
    if role in ASSISTANT_ROLES:
        return "assistant"
    return None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (_text_of(c) for c in content) if t)
    if isinstance(content, dict):
        if "parts" in content:
            return _text_of(content["parts"])
        for key in ("text", "content", "value", "message"):
            if key in content:
                return _text_of(content[key])
    return ""


def parse_text_memories(text: str) -> list[str]:
    facts = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "```")):
            continue
        line = re.sub(r"^([-*•·]|\d+[.)])\s+", "", line).strip()
        if len(line) >= 3:
            facts.append(line[:1000])
    return facts


def _chatgpt(conv: dict[str, Any]) -> Conversation:
    nodes = [n.get("message") for n in (conv.get("mapping") or {}).values() if isinstance(n, dict) and n.get("message")]
    nodes.sort(key=lambda m: m.get("create_time") or 0)
    msgs = []
    for m in nodes:
        role = _norm_role((m.get("author") or {}).get("role"))
        text = _text_of(m.get("content"))
        if role and text.strip():
            msgs.append((role, text.strip()))
    return Conversation(str(conv.get("title") or "Rozmowa ChatGPT"), msgs)


def _claude(conv: dict[str, Any]) -> Conversation:
    msgs = []
    for m in conv.get("chat_messages") or []:
        role = _norm_role(m.get("sender"))
        text = m.get("text") or _text_of(m.get("content"))
        if role and str(text).strip():
            msgs.append((role, str(text).strip()))
    return Conversation(str(conv.get("name") or conv.get("title") or "Rozmowa Claude"), msgs)


def _generic_messages(obj: Any, out: list[Conversation], title: str = "", depth: int = 0) -> None:
    """Find lists of message-like dicts anywhere in a JSON structure."""
    if depth > 8:
        return
    if isinstance(obj, dict):
        title = str(obj.get("title") or obj.get("name") or obj.get("subject") or title)
        for value in obj.values():
            _generic_messages(value, out, title, depth + 1)
    elif isinstance(obj, list):
        msgs = []
        for item in obj:
            if isinstance(item, dict):
                role = _norm_role(item.get("role") or item.get("author") or item.get("sender") or item.get("from"))
                text = _text_of(item.get("content") if "content" in item else item.get("text") or item.get("message"))
                if role and text.strip():
                    msgs.append((role, text.strip()))
        if msgs and len(msgs) >= max(1, len(obj) // 3):
            out.append(Conversation(title or "Rozmowa", msgs))
        else:
            for item in obj:
                _generic_messages(item, out, title, depth + 1)


def parse_json(data: Any, source_hint: str = "") -> ParsedExport:
    result = ParsedExport(source=source_hint or "json")
    if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
        result.facts = [x.strip()[:1000] for x in data if x.strip()]
        result.source = source_hint or "lista wspomnień"
        return result
    items = data if isinstance(data, list) else (data.get("conversations") or data.get("memories") or data.get("data")
                                                if isinstance(data, dict) else None)
    if isinstance(items, list) and items and isinstance(items[0], dict):
        first = items[0]
        if "mapping" in first:
            result.source = "ChatGPT"
            result.conversations = [c for c in (_chatgpt(x) for x in items if isinstance(x, dict)) if c.messages]
            return result
        if "chat_messages" in first:
            result.source = "Claude"
            result.conversations = [c for c in (_claude(x) for x in items if isinstance(x, dict)) if c.messages]
            return result
        if "header" in first and "time" in first and "title" in first:  # Google Takeout My Activity
            result.source = "Gemini (Google Takeout)"
            for x in items:
                title = str(x.get("title") or "")
                if "gemini" in str(x.get("header", "")).lower() or title.lower().startswith(("prompted", "zapytano")):
                    prompt = re.sub(r"^(Prompted|Zapytano o|Zapytano)\s*", "", title).strip()
                    if prompt:
                        result.conversations.append(Conversation(prompt[:60], [("user", prompt)]))
            return result
        memory_keys = ("memory", "content", "text", "value", "fact")
        if all(not any(k in x for k in ("role", "author", "sender", "messages", "mapping")) for x in items[:20]) and any(
            any(k in x for k in memory_keys) for x in items[:20]
        ):
            result.source = source_hint or "lista wspomnień"
            for x in items:
                text = next((x[k] for k in memory_keys if isinstance(x.get(k), str)), "")
                if text.strip():
                    result.facts.append(text.strip()[:1000])
            return result
    convs: list[Conversation] = []
    _generic_messages(data, convs)
    result.conversations = convs
    result.source = source_hint or "eksport JSON"
    return result


def parse_export(data: bytes, filename: str = "") -> ParsedExport:
    """Auto-detect and parse an export (zip, JSON or text)."""
    name = filename.lower()
    if data[:4] == b"PK\x03\x04" or name.endswith(".zip"):
        return _parse_zip(data)
    text = data.decode("utf-8-sig", errors="replace")
    stripped = text.lstrip()
    if stripped[:1] in "[{":
        try:
            return parse_json(json.loads(stripped), _hint(name))
        except json.JSONDecodeError:
            pass
    return ParsedExport(source=_hint(name) or "tekst", facts=parse_text_memories(text))


def _hint(name: str) -> str:
    for key, label in (("chatgpt", "ChatGPT"), ("openai", "ChatGPT"), ("claude", "Claude"), ("gemini", "Gemini"),
                       ("grok", "Grok"), ("deepseek", "DeepSeek"), ("mistral", "Mistral"), ("lechat", "Mistral")):
        if key in name:
            return label
    return ""


def _parse_zip(data: bytes) -> ParsedExport:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Uszkodzony plik ZIP") from exc
    total = sum(i.file_size for i in zf.infolist())
    if total > MAX_UNZIPPED:
        raise ValueError("Archiwum jest za duże po rozpakowaniu")
    merged = ParsedExport(source="")
    candidates = [i for i in zf.infolist() if not i.is_dir() and i.filename.lower().endswith((".json", ".txt", ".md"))]
    # Prefer the conversation/memory files; skip user/account metadata that holds no memories.
    candidates.sort(key=lambda i: (0 if re.search(r"conversation|memor|chat|activity", i.filename, re.IGNORECASE) else 1, i.filename))
    for info in candidates:
        if re.search(r"(^|/)(user|users|account|shared_conversations|message_feedback)\.json$", info.filename, re.IGNORECASE):
            continue
        part = parse_export(zf.read(info), info.filename)
        merged.conversations += part.conversations
        merged.facts += part.facts
        if part.source and part.source not in merged.source.split(", "):
            merged.source = f"{merged.source}, {part.source}" if merged.source else part.source
    merged.source = merged.source or "ZIP"
    return merged
