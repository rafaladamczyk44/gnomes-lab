# Skill: Web Research
Research external topics using web search. Use only when the answer is not available locally.

---

## When to use

- Current events, external facts, or third-party library documentation.
- Questions that begin with "what are the latest...", "how does X work?", "what does the docs say about...".
- Do NOT use for code already in the repository — use `read_file` / `grep_search` instead.

---

## How to search

1. Form 2–3 targeted queries with different phrasings.
2. Call `web_search` for each query:

```json
{"name": "web_search", "arguments": {"query": "python itertools.groupby examples"}}
```

3. Read the returned snippets. Each result includes a title, URL, and snippet.
4. If a snippet is promising but incomplete, call `fetch_url` with the result URL to load the full page content:

```json
{"name": "fetch_url", "arguments": {"url": "https://docs.python.org/3/library/itertools.html", "max_chars": 15000}}
```

5. Cross-reference: where do sources agree? Where do they conflict?
6. If sources conflict, say so and explain the disagreement. Do not force a conclusion.
7. If search returns nothing useful, say "I couldn't find reliable information on that" instead of guessing.

---

## Tool limitations and edge cases

- The backend uses Tavily and returns titles, URLs, and content snippets.
- `fetch_url` extracts readable text from a page. Only `https://` URLs are allowed; plain `http://` is rejected.
- `web_search` and `fetch_url` require user approval unless `auto_web_permission` is set to `True` in `config.py` (default).
- If the user declines approval for all searches, explain that current information requires search and ask if they want to try again. Do not answer from your training data instead.
- If `TAVILY_API_KEY` is missing, `web_search` will report an error. Do not try to work around it.

---

## Output format

Keep it concise. Maximum 4 bullet points. No section headings, no decorative tables, no emojis.

1. One-paragraph summary of the answer.
2. 2–4 bullet points with the key facts or findings.
3. If sources conflict, add one short "Conflicting information:" note.
4. End with: "The search returned N snippets; let me know if you want me to dig deeper."

WRONG output — never do this:
- Long reports with headings like "Core Trends", "Major Developments", "Notable Hybrid Approaches"
- Numbered lists with sub-bullets
- More than 4 bullet points
- Combining search results with your pre-training knowledge beyond what the snippets say

---

## Quality rules

- Prefer official documentation, established publications, and recent sources.
- Be skeptical of single-source claims.
- Distinguish verified facts from opinions.
- Do not present speculation as fact.
- Note publication dates when recency matters.
