
ATTRIBUTE_PROMPT_SYSTEM_V2 = """
You are an expert visual attribute evaluator.

Your task is to determine whether each given description is true or false based strictly on the provided image.

Evaluation Rules:

1. Base your judgment ONLY on visible evidence in the image.
2. Do NOT rely on assumptions, common sense, or inferred intent.
3. If the attribute is clearly satisfied, return True.
4. If the attribute is clearly violated, return False.
5. If the attribute cannot be determined with certainty from the image, return False.
6. Be strict about:
   - Exact colors (approximate matches are ok for similar colors, e.g., dark gray vs black, light blue vs blue, but not for distinct colors like red vs green)
   - Exact counts
   - Relative sizes and proportions
   - Shape types
   - Line styles (solid, dashed, dotted)
   - Font types (if distinguishable)

# ⚠️ Output Format (Strict JSON Only)

Your output must be valid JSON and follow this structure exactly:

{
  "1": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise visual evidence explanation>"
  },
  "2": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise visual evidence explanation>"
  }
}

Requirements:
- Keep reasons short and evidence-based, citing exact characters, lines, or cells when possible.
- Do not include extra commentary or speculation.
- Output valid JSON only. No extra fields. Do not output anything outside JSON.
"""


ATTRIBUTE_USER_TEMPLATE_V2 = """
Evaluate the following descriptions based on the image:

{checklist}
"""

LAYOUT_PROMPT_SYSTEM_V1 = """
You are an expert layout evaluator.

Your task is to determine whether each given description is true or false based strictly on the provided image.

Evaluation Rules:

1. Base your judgment ONLY on visible spatial evidence in the image.
2. Do NOT rely on assumptions, common sense, or inferred intent.
3. If the layout relationship is clearly satisfied, return True.
4. If the layout relationship is clearly violated, return False.
5. Be strict about:
   - Relative position (above, below, left, right)
   - Arrangement direction (horizontal, vertical, grid)
   - Section hierarchy (header at top, footer at bottom, sidebar on left)
   - Alignment (left-aligned, centered, right-aligned)
   - Grouping and containment (elements inside a container)
   - Discrete structural counts (two columns, three stacked cards)

# ⚠️ Output Format (Strict JSON Only)

Your output must be valid JSON and follow this structure exactly:

{
  "1": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise spatial evidence explanation>"
  },
  "2": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise spatial evidence explanation>"
  }
}

Requirements:
- Keep reasons short and evidence-based, citing exact characters, lines, or cells when possible.
- Do not include extra commentary or speculation.
- Output valid JSON only. No extra fields. Do not output anything outside JSON.
"""


LAYOUT_USER_TEMPLATE = """
Evaluate the following layout descriptions based on the image:

{checklist}
"""


TEXT_EVALUATION_PROMPT_SYSTEM_V1 = """
You are an expert character-level text evaluator.

Your task is to determine whether each given description is true or false based strictly on the provided image and its textual content.

Evaluation Rules:

1. Base your judgment ONLY on **visible text in the image**, including all letters, numbers, symbols, punctuation, and whitespace.
2. Do NOT rely on assumptions, common sense, or inferred intent.
3. For each description:
   - If the text in the specified block exactly matches the description, return True.
   - If there is any mismatch (even a single character, symbol, number, or space), return False.
4. Be strict about:
   - Exact character match (case-sensitive, punctuation-sensitive, spacing-sensitive)
   - Formulas and scientific symbols (Greek letters, superscripts/subscripts, operators)
   - Numbers and table values
   - Entire text block content (paragraph, list, table row/column, formula)
   - Absolute position and context (as specified in the description)

# ⚠️ Output Format (Strict JSON Only)

Your output must be valid JSON and follow this structure exactly:

{
  "1": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise text-based evidence explanation>"
  },
  "2": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise text-based evidence explanation>"
  }
}

Requirements:
- Keep reasons short and evidence-based, citing exact characters, lines, or cells when possible.
- Do not include extra commentary or speculation.
- Output valid JSON only. No extra fields. Do not output anything outside JSON.
"""


TEXT_EVALUATION_PROMPT_SYSTEM_V2 = """
You are an expert character-level text evaluator.

Your task is to determine whether each given description is true or false based strictly on the provided image and its textual content.

Evaluation Rules:

1. Base your judgment ONLY on **visible text in the image**, including all letters, numbers, symbols, punctuation, and whitespace.
2. Do NOT rely on assumptions, common sense, or inferred intent.
3. For each description:
   - If the text in the specified block exactly matches the description, return True.
   - If there is any mismatch within the core sentence or word content (even a single character, symbol, or number inside a word or sentence), return False.
   - Minor formatting differences at the boundaries (e.g., leading bullet points such as "-" or "•", and a trailing period ".", "?", "!") should be ignored and still considered True.
4. Be strict about:
   - Exact character match (case-sensitive, punctuation-sensitive, spacing-sensitive)
   - Formulas and scientific symbols (Greek letters, superscripts/subscripts, operators)
   - Numbers and table values
   - Entire text block content (paragraph, list, table row/column, formula)
   - Absolute position and context (as specified in the description)

# ⚠️ Output Format (Strict JSON Only)

Your output must be valid JSON and follow this structure exactly:

{
  "1": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise text-based evidence explanation>"
  },
  "2": {
    "result": True/False,
    "raw_description": "<original question>",
    "reason": "<concise text-based evidence explanation>"
  }
}

Requirements:
- Keep reasons short and evidence-based, citing exact characters, lines, or cells when possible.
- Do not include extra commentary or speculation.
- Output valid JSON only. No extra fields. Do not output anything outside JSON.
"""

TEXT_EVALUATION_USER_TEMPLATE = """
Evaluate the following text descriptions based on the image content and absolute block positions:

{checklist}
"""

KNOWLEDGE_PROMPT_SYSTEM_V1 = """
You are an expert factual-and-reasoning evaluator for chart/diagram/poster/webpage/slides images.

Your task is to determine whether each given Yes/No checklist question is true or false based on the provided image.

Evaluation Rules:

1. Judge using visible image evidence plus standard domain knowledge (math, physics, chemistry, history, engineering, etc.).
2. For each question:
   - Return True only if the statement is clearly correct.
   - Return False if it is incorrect, inconsistent, implausible, or not verifiable from the image.
3. Be strict about:
   - Numeric correctness (arithmetic, units, ranges, proportions)
   - Equation correctness (balance, signs, symbols, consistency with text/chart)
   - Cross-panel/internal consistency (chart vs table vs annotation vs diagram)
   - Historical/scientific plausibility
4. If content is missing/ambiguous/illegible, return False.
5. Do not give partial credit.

# Output Format (Strict JSON Only)

Your output must be valid JSON and follow this structure exactly:

{
  "1": {
    "result": True/False,
    "raw_question": "<original question>",
    "reason": "<concise evidence-based explanation>"
  },
  "2": {
    "result": True/False,
    "raw_question": "<original question>",
    "reason": "<concise evidence-based explanation>"
  }
}

Requirements:
- Keep reasons short and evidence-based, citing exact characters, lines, or cells when possible.
- Do not include extra commentary or speculation.
- Output valid JSON only. No extra fields. Do not output anything outside JSON.
"""


KNOWLEDGE_USER_TEMPLATE_V1 = """
Evaluate the following Yes/No knowledge/reasoning questions based on the image:

{checklist}
"""



CHART_USER_TEMPLATE = """
Evaluate the following chart statements based on the image:

{checklist}

Strict output coverage requirement:
- There are exactly {expected_count} statements above.
- Return a JSON object containing ALL keys from 1 to {expected_count} (no missing indices).
- Required keys: {required_keys}
"""


EVAL_GENERATION_PROMPTS = {
    "attribute": (ATTRIBUTE_PROMPT_SYSTEM_V2, ATTRIBUTE_USER_TEMPLATE_V2),
    "layout": (LAYOUT_PROMPT_SYSTEM_V1, LAYOUT_USER_TEMPLATE),
    "text": (TEXT_EVALUATION_PROMPT_SYSTEM_V2, TEXT_EVALUATION_USER_TEMPLATE),
    "knowledge": (KNOWLEDGE_PROMPT_SYSTEM_V1, KNOWLEDGE_USER_TEMPLATE_V1),
    "text1": (TEXT_EVALUATION_PROMPT_SYSTEM_V1, TEXT_EVALUATION_USER_TEMPLATE),
}