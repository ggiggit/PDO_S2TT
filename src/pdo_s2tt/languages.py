"""Prompts used for the five released English-to-X directions."""

LANGUAGE_NAMES = {
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "fr": "French",
}


def instructions(target_code: str) -> tuple[str, str, str]:
    code = target_code.strip().lower()
    if code not in LANGUAGE_NAMES:
        supported = ", ".join(LANGUAGE_NAMES)
        raise ValueError(f"unsupported target language {target_code!r}; choose {supported}")
    target = LANGUAGE_NAMES[code]
    system = (
        f"You are a professional English to {target} simultaneous translator. "
        f"Translate only the current English unit and output {target} text only, "
        "except when the user explicitly states that no English source text is available. "
        "The English text is produced by streaming automatic speech recognition and may "
        "contain missing words, substitutions, punctuation errors, or errors in names and "
        "numbers. Use the speech evidence and context to preserve every supported clause, "
        "name, number, date, time, unit, negation, and relation exactly once. Correct an "
        "error only when the evidence supports it; do not omit, repeat, or invent information."
    )
    user = (
        f"Translate all English speech heard so far into {target}. Return only the complete "
        f"cumulative {target} translation. If no {target} word is supported yet, end this "
        "assistant turn without text:\n"
    )
    return target, system, user
