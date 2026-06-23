"""Output parsing for the Antigravity CLI (agy).

agy ``--print`` returns the final answer as plain text (occasionally wrapped
in a small JSON envelope), so only batch extraction is needed.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def parse_antigravity_json(raw: str) -> str:
    """Extract result text from Antigravity CLI ``--print`` output.

    Tries to parse as JSON; falls back to raw text truncated to 2000 chars.
    """
    if not raw:
        return ""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Try common content keys
            for key in ("content", "result", "text", "message"):
                val = parsed.get(key)
                if isinstance(val, str) and val:
                    return val
            return str(parsed)
        return str(parsed)
    except json.JSONDecodeError:
        return raw[:2000]
