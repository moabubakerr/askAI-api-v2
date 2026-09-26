# `POST /ask` — frontend integration brief

The user picks who answers their question: SCAI, Oxford Economics, or both.
This is a new endpoint alongside `/chat`; `/chat` is unchanged and still works.

## Request

Same body as `/chat`, plus two fields.

```jsonc
{
  "message": "What was Qatar's GDP growth in 2024?",
  "session_id": "abc-123",      // same session id you already use for /chat
  "source": "combined",         // "scai" | "oxford" | "combined"   (default "scai")
  "oxford_mode": "auto"         // "auto" | "data" | "analysis" | "both"  (default "auto")
}
```

`source` is the only field you must add. Leave `oxford_mode` alone unless you
build an advanced control — `auto` asks our router model to read the question
and pick the right Oxford tool.

`/ask` and `/chat` share one session, so a user can switch sources mid-conversation
and follow-ups still resolve.

## Response (HTTP 200)

```jsonc
{
  "source": "combined",
  "message_id": "6f8776e8-…",   // the SCAI answer's id when present, else Oxford's
  "answer": "## SCAI\n\n…\n\n---\n\n## Oxford Economics\n\n…",
  "scai":   { /* SourceAnswer, null when source=oxford   */ },
  "oxford": { /* SourceAnswer, null when source=scai     */ }
}
```

Each `SourceAnswer`:

```jsonc
{
  "source": "oxford",
  "ok": true,
  "answer": "markdown, including tables",
  "error": null,                 // string when ok is false
  "latency_ms": 18014,
  "message_id": "b340e55e-…",    // rate THIS half via POST /feedback
  "verified": false,             // see below — always false for Oxford
  "readable": false,             // SCAI only: show the "Read this for me" button
  "facts_payload": {},           // SCAI only
  "chart": null,                 // SCAI only
  "tools_used": ["EconomicData"] // Oxford only
}
```

### Render the blocks, not `answer`

`answer` is a pre-rendered convenience for a caller with one text area. For the
comparison UI, render `scai` and `oxford` as two separate panels and ignore
`answer` — do not split it back apart on `---`.

The two answers are deliberately **not merged into one voice**, and must not be
merged in the UI either. They are two houses answering the same question, and
the user needs to see who said what. Expect them to disagree; that is the
feature, not a bug to smooth over.

### `verified` is a label you must surface

`verified: true` means every figure in that text was checked against the rows
our pipeline retrieved. Oxford answers are **always** `verified: false` — their
prose is composed over data we do not hold, so there is nothing to check it
against.

Show this difference. A badge on the SCAI panel, or a note on the Oxford panel
("Oxford Economics' own figures, not verified against SCAI data"). Do not style
the two panels identically with no indication of which carries our guarantee.

Oxford's text is passed through byte for byte, including figures that look
wrong. Do not filter or "correct" anything client-side.

### Markdown

Both sources return markdown, and Oxford leans on **tables** heavily
(region / year / variable / value / units). Whatever renderer you use must
handle GFM tables, or their answers will come through as pipe soup.

## Latency — the thing that will shape your UI

| source | typical |
|---|---|
| `scai` | as today |
| `oxford` | **~20s cold**, ~0.5s for a repeat of the same question (they cache) |
| `combined` | ~max of the two — both run in parallel server-side |

Twenty seconds of blank screen is not acceptable. `/ask` has **no streaming or
progress events** (unlike `/chat`, which supports `Accept: text/event-stream`).
So you need a determinate-feeling wait: per-panel skeletons, an "Asking Oxford
Economics…" label, and a client timeout of **at least 130s** — the server's own
ceiling is 120s.

In `combined`, both halves arrive in one response, so you cannot show SCAI's
answer early. If that matters, say so and we will add SSE to `/ask`.

## Errors

**One half fails, the other works** → still HTTP 200. Check `ok` per panel and
render `error` in place of that panel's answer. Never discard a good answer
because its neighbour failed.

**HTTP 503** — Oxford is not configured on this deployment:

```jsonc
{ "detail": { "ok": false, "message": "Oxford Economics is not configured on this deployment." } }
```

Treat this as "hide the Oxford and Combined options". Probe it once at startup
with a throwaway `source=oxford` request, or ask us for a capability flag on
`/health` — say which you prefer.

**HTTP 502** — `source=oxford` and Oxford failed, or `combined` and *both*
failed. Same `detail` shape. Show `detail.message`; it is written for a reader.

## Feedback

`POST /feedback` is unchanged. Each panel carries its own `message_id`, so a
user can rate the two answers independently — worth building, since comparing
the two houses is the point of the trial and per-source ratings are the
cleanest signal we will get about which one users prefer.

## Privacy — please put this in front of the user

`source=oxford` and `source=combined` send the user's question **verbatim to
Oxford Economics' cloud**. Everything else in this product stays on-premises.

That makes the source selector a data-egress decision, not a cosmetic one. It
should be an explicit, visible choice — not a remembered default that silently
persists across sessions, and not preselected. The wording is the product
team's call, but the user should be able to tell that picking Oxford or
Combined sends their question outside.

## Try it

```bash
curl -s -X POST localhost:18000/ask -H 'Content-Type: application/json' \
  -d '{"message":"What was Qatar'\''s GDP growth in 2024?","session_id":"demo","source":"combined"}'
```

Full schema is in the OpenAPI docs at `/docs`.
