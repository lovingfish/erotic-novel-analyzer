def build_meta_prompt(content: str) -> str:
    return f"""
You are a professional literary analyst specializing in adult fiction.

## Task
Extract novel metadata and write a 200-300 Chinese character summary.

## Output JSON ONLY
- Return ONLY one JSON object. No extra text.
- Use double quotes for keys/strings.
- Required keys: novel_info, summary.

### Schema
{{
  \"novel_info\": {{
    \"world_setting\": \"\",
    \"world_tags\": [],
    \"chapter_count\": 0,
    \"is_completed\": false,
    \"completion_note\": \"\"
  }},
  \"summary\": \"\"
}}

## Novel Content
{content}
"""


def build_core_prompt(content: str) -> str:
    return f"""
As a professional literary analyst specializing in adult fiction, extract ONLY characters who engage in sexual activities and their sexual relationships.

## Requirements
- Characters: ONLY those with sexual activity.
- Gender MUST be \"male\" or \"female\" (lowercase English). No other values.
- Include: name, gender, identity, personality, sexual_preferences.
- For female characters: lewdness_score (1-100 int) and lewdness_analysis (required).
- First-person narrator (\"我\") must be included if involved; infer name/alias if possible, else use \"我\".
- Map every sexual relationship with from/to/type/start_way/description.

## Output JSON ONLY
{{
  \"characters\": [],
  \"relationships\": []
}}

## Novel Content
{content}
"""


def build_scenes_prompt(content: str, allowed_names_json: str, relationships_json: str) -> str:
    return f"""
You are analyzing intimacy scenes and relationship evolution. Use ONLY the provided character names.

### Allowed character names (MUST use exactly):
{allowed_names_json}

### Known relationships (reference only):
{relationships_json}

## Output JSON ONLY
{{
  \"first_sex_scenes\": [],
  \"sex_scenes\": {{
    \"total_count\": 0,
    \"scenes\": []
  }},
  \"evolution\": []
}}

## Rules
- participants must be from allowed names; do NOT invent new names.
- chapter/location/description MUST be non-empty strings. If unknown, write \"未知\" or \"未提及\".
- evolution.chapter/stage/description MUST be non-empty strings. If unknown, write \"未知\" or \"未提及\".
- sex_scenes.total_count must be an integer.

## Novel Content
{content}
"""


def build_thunderzones_prompt(content: str, allowed_names_json: str, relationships_json: str) -> str:
    return f"""
You are detecting thunderzones (reader deal-breakers) in an adult novel. Use ONLY the provided character names.

### Allowed character names (MUST use exactly):
{allowed_names_json}

### Known relationships (reference only):
{relationships_json}

## Thunderzone types
- 绿帽/Cuckold
- NTR (Netorare)
- 女性舔狗
- 恶堕
- 其他

Severity must be: 高 / 中 / 低.

## Output JSON ONLY
{{
  \"thunderzones\": [],
  \"thunderzone_summary\": \"\"
}}

## Rules
- involved_characters must be from allowed names; do NOT invent new names.
- description MUST be non-empty. If unknown, write \"未知\" or \"未提及\".
- chapter_location is OPTIONAL. If unknown, you can leave it empty.
- relationship_context can be empty, but if provided must be a string.
- If no thunderzones, return empty array and a short summary.
- Use EXACT keys: type, severity, description, involved_characters, chapter_location, relationship_context.
- If you cannot provide a non-empty description for an item, DO NOT include that item.

## Novel Content
{content}
"""
