# SCAI Economic Data Assistant — On-Prem Architecture

## 1. Components

| Component        | Tech                                  | Notes |
|-------------------|----------------------------------------|-------|
| LLM serving       | vLLM + Qwen2.5-72B-Instruct (or Llama-3.3-70B) | Runs on GPU host, exposes OpenAI-compatible API |
| Router model      | Qwen2.5-7B-Instruct (same or separate vLLM instance) | Fast intent classification |
| Structured data   | PostgreSQL                             | SCAI indicators, read-only role for the app |
| Unstructured data | Chroma (local, persisted to disk)      | For SCAI PDF reports/publications later |
| Orchestration     | LangGraph                              | Deterministic state machine, not free-form agent loop |
| API               | FastAPI                                | Single network-facing entrypoint |

Nothing in this stack calls out to the public internet at runtime.

## 2. Starting the LLM server (example, on the GPU machine)

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --port 8001 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.90
```

Run a second, smaller instance for the router model on another port if you
have GPU headroom, or just reuse the 72B for both (simpler, slower).

## 3. Database setup

> **Running under Docker (§4a)? You do not run any of the commands in this
> section by hand, and none of them happen at image-build time.**
>
> - `etl/schema.sql` is mounted into the Postgres container's
>   `docker-entrypoint-initdb.d/` and applied automatically the first time the
>   `pgdata` volume initialises.
> - The `scai_ro` role is created right after it by `docker/initdb/02-roles.sh`,
>   in the same first-boot pass. Ordering matters —
>   `GRANT SELECT ON ALL TABLES` only covers tables that exist at grant time,
>   which is why the files are numbered `01-`/`02-`.
> - The data load is `docker compose run --rm etl`, against the running
>   container, **after** the stack is up. It is a runtime step, not a build
>   step: the CSVs are bind-mounted read-only and `data/` is in `.dockerignore`,
>   so the image carries no SCAI data and reloading never needs a rebuild.
> - Nothing runs on your workstation. No local Postgres, no local `pip install`.
>
> Both init scripts run **once**, on an empty volume. Editing `schema.sql` later
> changes nothing until you `docker compose down -v` (which destroys the data) or
> apply the change by hand with the `psql` that ships in the app image.
>
> The section below is the equivalent by-hand procedure, for a Postgres you
> manage yourself.

Create the schema (built from actual inspection of the SCAI CSV exports —
see `etl/schema.sql` for full column-level detail and comments):

```bash
createdb scai_indicators
psql scai_indicators -f etl/schema.sql
```

Load the data (needs a write-capable role — create one separately from the
app's read-only role below):

```bash
pip install -r requirements.txt
python etl/load_data.py \
  --csv-dir /path/to/scai/csv/exports \
  --dsn "postgresql://scai_rw:password@localhost:5432/scai_indicators"
```

This loads, in order: countries, intervals, indicators (531, merged from the
Item_2 catalog + P01 published layer), indicator_details (644, merged from
Item_4 + P02), indicator_values (11,303 raw data points), published_data_points
(8,127 curated points with precomputed MoM/QoQ/YoY), indicator_analysis (2,257
merged analyst commentary rows), benchmark_countries, and the CMS tables
(champions, sectors, general_entities, articles) with base64-decoded and
mojibake-fixed text. All of this has been validated against the real files
you provided — see the "What the ETL actually does" section below.

Then create the app's read-only role:

```sql
CREATE ROLE scai_ro WITH LOGIN PASSWORD 'change_me';
GRANT CONNECT ON DATABASE scai_indicators TO scai_ro;
GRANT USAGE ON SCHEMA public TO scai_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO scai_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO scai_ro;
```

The app connects with `scai_ro` only — it physically cannot write, regardless
of what SQL the LLM generates. Point `POSTGRES_DSN` in `app/core/config.py`
(or your `.env`) at this role, never at the `scai_rw` role used for loading.

### What the ETL actually does (validated against your real files)

- **Two data layers, merged:** the `Item_*` files are the full working dataset
  (531 indicators); the `P0*` "Published" files are a curated, reviewed subset
  (189 indicators) with extra metadata (unit, polarity, precomputed period-
  over-period % change). The loader keeps both, with `is_published` flags and
  FK links, so the SQL agent can use the richer published data when available
  and fall back to raw data otherwise.
- **Mojibake fix:** several free-text fields (e.g. analyst commentary) were
  double-encoded (UTF-8 bytes mis-decoded as cp1252, re-saved as UTF-8),
  producing garbage like `QCBâ€™s`. Fixed via an `encode('cp1252').decode('utf-8')`
  round-trip — confirmed against your actual data.
- **Base64-decoded rich text:** `Sectors`/`General_Entities` description fields
  are inconsistently stored as base64-encoded HTML in some rows and plain text
  in others. The loader detects and decodes per-row, then strips HTML tags for
  clean LLM grounding text.
- **Period normalization:** periods appear as `'YYYY'`, `'YYYY-Q#'`, or
  `'YYYY-MM'`. All are parsed into a proper `DATE` (first day of period) plus
  a `granularity` label, verified against all 11,303 + 8,127 real rows with a
  0% unparseable rate.
- **Referential-integrity gaps in the source export — rows are dropped, loudly.**
  The counts quoted just above are the raw CSV row counts. Loading them against
  the real FK constraints in `schema.sql` fails, because the export is not
  internally consistent. What actually lands, and why:

  | Table | CSV rows | Loaded | Dropped |
  |---|---|---|---|
  | `indicators` | 531 | 470 | 61 with a blank `NameEN` *and* blank `NameAR` |
  | `indicator_details` | 644 | 583 | 61 whose `IndicatorId` (52 distinct) is absent from `Item_2` |
  | `indicator_values` | 11,303 | 9,497 | 1,806 sitting under those dropped details |

  The nameless indicators are unusable regardless: the resolver matches on
  `name_en`, so an indicator with no name can never be reached by a question nor
  cited in an answer. The orphaned details look like indicators deleted upstream
  without cascading.

  **The curated layer is completely unaffected** — 189 published indicators, 289
  published details and all 8,127 `published_data_points` load intact, and none
  of the dropped rows are published (0 appear in `P01`/`P02`). Everything lost is
  unpublished working data. The loader prints each skip count on every run.

  Worth raising with SCAI: the 52 missing parent indicators are the one item
  here that could represent real data the chatbot should have. If they send
  those rows, ~1,806 additional raw data points come back with them.

- **Known gap:** `StageOfMonitoringId`/`StatusId`/`WorkflowId` on
  `Sectors`/`General_Entities` are opaque GUIDs — no lookup table was included
  in what you sent. They're loaded as-is; the chatbot won't be able to answer
  "what stage is X in" by a human-readable name unless you send that lookup
  table.

### Second batch of files (chart/dashboard metadata + reference tables)

- **`countries` now sourced from `P14_Ref_Countries`** instead of the earlier
  `Item_6_Countries` — same row count but adds ISO country code + lat/long,
  which the chart layer needs for the "Geo Map" chart type SCAI itself uses.
- **`indicator_dashboards` (from `P09_Published_Dashboards`) is the highest-value
  addition here** — it's the bridge between the numeric indicator world and the
  CMS/sector world. Verified: 210 rows link to `sectors`, 128 link to
  `general_entities`, joined correctly by `entity_classification_name`. This is
  what makes "which indicators track the Tourism sector" answerable at all —
  without it, indicators and sectors were two disconnected islands of data.
- **`lookups` (from `P12_Ref_Lookups_Common`)** and **`priority_types`
  (from `P13`)** are small generic reference tables — mainly useful for
  resolving IDs to labels, not something users ask about directly.
- **Chart metadata tables** (`chart_sub_indicators`, `chart_alternate_views`,
  `additional_charts`, `additional_chart_countries` — from P07a/P07b/P08/P08a)
  are loaded but low-priority for Q&A; they exist so the Chart Agent can
  eventually match SCAI's own chosen chart types/views rather than guessing.
- **`P10_Published_Mappings` and `P11_Published_ValidationChecks` were empty**
  (zero bytes) — not loaded. If these get populated later, let me know and
  I'll add loaders.
- **`P16_Base_Indicators_MappingTargets` and `P17_Base_IndicatorDetails_KeyMap`
  were skipped** — verified their IDs map 1:1 onto the existing published
  indicator set (`P16` → 189/189 matched `Item_2`, `P17` → 289/289 matched
  `Item_4`), so they're a re-confirmation of data already loaded via P01/P02,
  not new information. Flag it if you know of a field in there we're missing.

## 4. Running the app

### 4a. Docker on the VM (the supported path)

The image is built on the VM, from the repo, next to the models — there is no
registry push and no `docker save`/`load` step.

**Step 0 — clone. The CSVs come with it.**

`data/` is committed to this repository, so a clone is self-contained and no
separate file transfer is needed before the ETL can run:

```bash
git clone https://github.com/moabubakerr/askAI-api-v2.git ~/scai-chatbot
cd ~/scai-chatbot
```

The CSVs are still kept out of the Docker *image* (`data/` is in
`.dockerignore`) and bind-mounted read-only into the `etl` container at load
time — so the image carries no data, and reloading never needs a rebuild.

**Step 1 — find the model stack's network name.** This is the one value that
cannot be guessed: Compose namespaces networks by project, so the network your
`vllm`/`tei` services sit on is almost certainly `<project>_runtimes`, not a
bare `runtimes`.

```bash
docker network ls | grep -i runtimes
# e.g.  a1b2c3d4  models_runtimes  bridge  local   <- use this exact name
```

Confirm the model containers are really on it, and that the hostnames the app
will use actually resolve there:

```bash
docker network inspect <that-name> --format '{{range .Containers}}{{.Name}} {{end}}'
```

**Step 2 — configure and start:**

```bash
cd /opt/scai-chatbot
cp .env.example .env
${EDITOR:-vi} .env        # set both passwords + RUNTIMES_NETWORK from step 1

docker compose up -d --build     # builds the image on the VM, starts postgres + api
docker compose run --rm etl      # one-shot data load
```

**Step 3 — verify, in this order.** Each check isolates one dependency, so a
failure tells you which piece is wrong:

```bash
# a) the process is up (proves nothing else)
curl -s localhost:18000/health

# b) the database loaded and the read-only role can see it
docker compose exec api python -c \
  "from app.resolvers.indicator_resolver import _fetch_catalog; print(len(_fetch_catalog()), 'indicators visible')"
# expect: 583 indicators visible

# c) the models are reachable from inside the app container
docker compose exec api python -c \
  "from openai import OpenAI; print([m.id for m in OpenAI(base_url='http://vllm:8000/v1', api_key='x').models.list()])"
# expect: ['qwen72b']
docker compose exec api curl -s -X POST http://tei:80/v1/embeddings \
  -H 'Content-Type: application/json' -d '{"input":"inflation","model":"bge-m3"}' | head -c 80

# d) end to end — the first one is slow, see the cold-start note below
curl -s -X POST localhost:18000/chat -H 'Content-Type: application/json' \
  -d '{"message":"What can you do?"}'
```

Check (d) with "What can you do?" first on purpose: it is the one data-backed
path that does **not** go through the embedding resolver, so it separates
"models unreachable" from "embedding cold start is still running".

### Troubleshooting

| Symptom | Cause |
|---|---|
| `network ... declared as external, but could not be found` at `up` | `RUNTIMES_NETWORK` doesn't match step 1, or the model stack is down. Start the models first — this compose file never creates that network. |
| First `/chat` hangs for minutes, later ones are fast | Expected. Embedding cold start — see below. |
| `404 model not found` from vLLM | Something reset `LLM_MODEL_NAME`/`ROUTER_MODEL_NAME` off the literal `qwen72b`. |
| `could not translate host name "vllm"` | The `api` container isn't on the model network — re-check step 1. |
| ETL: `relation "indicators" does not exist` | Postgres came up on a volume that predates `schema.sql` being mounted. `docker compose down -v` re-inits (destroys DB contents). |
| ETL: `permission denied` | `ETL_DSN` is pointing at `scai_ro`. It must use `POSTGRES_USER`. |

Logs: `docker compose logs -f api`.

`docker compose up` does **not** run the ETL — it lives behind the `etl` profile
because it `TRUNCATE`s every table before loading. It is a full reload, run on
purpose, not on every restart.

Smoke test:
```bash
curl -s localhost:18000/health
curl -s -X POST localhost:18000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What was Qatar real GDP growth in 2023?"}'
```

**How it attaches to the models.** vLLM and TEI are not defined in this
compose file — they already run in their own project on the `runtimes` network,
which this file joins as `external: true`. The `api` service sits on both that
network and a private `internal` one for Postgres. Consequences worth knowing:

- `runtimes` must already exist when you bring the stack up. If the model stack
  is down, `docker compose up` fails with "network runtimes declared as external,
  but could not be found" — start the models first.
- **The served model name is `qwen72b`** even though the weights are
  Qwen2.5-32B-Instruct. vLLM 404s on a mismatch, so `LLM_MODEL_NAME` and
  `ROUTER_MODEL_NAME` are both pinned to that exact string in
  `docker-compose.yml`. Do not "fix" the number.
- `ROUTER_MODEL_NAME` mattering is easy to miss: the config default is
  `Qwen2.5-7B-Instruct`, and against a single-model server that default would
  404 on every intent classification (`app/nlu/intent_agent.py`) and every
  verifier pass. Both now point at the one served model.
- TEI is reached at `http://tei:80/v1` — port 80 inside the network; `17800` is
  only the host-side publish and is not what the container uses.
- `max_model_len` is 16384 for prompt **and** completion together.
  `LLM_MAX_TOKENS` is set to 1500, leaving ~14.8k for the prompt.

**Cold start is slow, by design of the current code.** `app/resolvers/embeddings.py`
caches vectors in a process-local dict, and `embed_catalog()` embeds uncached
names one HTTP request at a time. After every `api` restart the first data
question therefore issues ~583 sequential embedding calls against a CPU-bound
TEI before it answers. Expect the first request to take minutes and later ones
to be fast. Fixing it properly means persisting the vectors (pgvector) or
batching the call — neither is done yet.

The API is published on `${API_PORT}` (default **18000**, not 8000, since 8000
is usually taken on a VM already running vLLM).

### 4b. Bare metal, without Docker

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

Note `requirements.txt` is missing an `httpx<0.28` pin that `openai==1.51.0`
needs — see `docker/requirements.runtime.txt`. Without it the app does not
import at all.

## 5. Agent flow

```
User question
   │
   ▼
[Router Agent] ── classifies intent (small/fast model)
   │
   ├── general_chat / out_of_scope → direct answer, END
   │
   └── data_lookup / trend_analysis / comparison / chart_request
          │
          ▼
   [Text-to-SQL Agent] ── generates SELECT, validates, executes (retries once on error)
          │
          ▼
   [Data Analyst Agent] ── writes NL answer FROM the retrieved rows only
          │
          ├── (if chart requested) → [Chart Agent] ── produces chart spec
          │
          ▼
   [Verifier Agent] ── checks every number in the draft traces to the data
          │
          ▼
   Final answer + optional chart spec → returned to user
```

## 6. Next steps once you share the real SCAI data

1. Replace `app/db/schema.py` with the actual table/column descriptions.
2. Load the Excel/CSV data into Postgres (a one-off ETL script — happy to write
   this once I see the real file structure).
3. If SCAI also publishes PDF reports, add a RAG agent (Chroma + embeddings
   model, e.g. `bge-m3` served locally) to handle narrative/qualitative
   questions the structured DB can't answer.
4. Add conversation memory (Postgres-backed session store) so follow-up
   questions like "and the year before that?" resolve correctly.
5. Add an eval set: 30–50 real questions with known-correct answers, run
   after every schema or prompt change to catch regressions.

---

# v2 Architecture — Response to SCAI QC Report

The tester feedback workbook (35 findings, 29 Major) showed a consistent
pattern in the previously-implemented "Ask AI" system: **the LLM was doing
arithmetic and free-form retrieval, and getting both wrong.** Growth rates,
min/max lookups, period differences, and country comparisons were computed
or selected by the LLM per-question, with no deterministic, testable logic
behind them.

**v2 removes the LLM from every numeric decision.** The old `app/core/graph.py`
(Text-to-SQL agent + LLM analyst doing math) is deprecated. The new pipeline
is `app/core/graph_v2.py`:

```
User question
   │
   ▼
[Intent Agent] (LLM — extraction ONLY, structured JSON, no numbers)
   │
   ▼
[Indicator Resolver]   deterministic, embedding-based semantic match
[Country/Group Resolver]   deterministic, GCC + alias handling
[Period/Frequency Resolver]   deterministic regex + fixed granularity rule
   │
   ▼
[Retriever]   fixed, parameterized SQL — no LLM-generated queries
   │
   ▼
[Compute Engine]   pure Python — latest/trend/diff/growth/min-max/compare/rank
                   (app/compute/engine.py — the ONLY place arithmetic happens)
   │
   ▼
[Composer Agent] (LLM — phrasing ONLY, forbidden from calculating/inventing)
   │
   ▼
[Numeric Verifier]   non-LLM regex check: every number in the answer must
                     trace to the facts payload, or fall back to a template
   │
   ▼
Final answer
```

## Findings → root cause → fix

| Finding(s) | Root cause | Fix in v2 |
|---|---|---|
| F-005, F-017 | LLM computed growth rate itself, wrong formula/data | `compute.growth_rate()` — fixed CAGR/simple formula, tested |
| F-008 | "Highest value" assumed = latest, not actually scanned | `compute.min_max()` — scans all retrieved rows |
| F-009, F-010, F-011 | LLM couldn't/wouldn't do arithmetic reliably | `compute.difference()`, `compute.period_to_period_change()` |
| F-022–026, F-031 | QoQ/YoY/MoM confused, or recomputed instead of using SCAI's own vetted figures | `compute.preferred_change_field()` — always prefers `published_data_points`' precomputed columns over re-deriving |
| F-003, F-012 | Wrong indicator silently substituted (inactive VC-deals indicator; Nominal vs Real GDP) | `indicator_resolver.py` — confidence threshold, ambiguity-gap check, explicit contrastive-term contradiction check (nominal/real, gross/net, etc.) |
| F-014 | Paraphrase ("tourists arrived") didn't match catalog string | Embedding-based semantic match (`embeddings.py`) — tested: string similarity alone scores this pair 0.167, unusable; embeddings solve the actual synonym problem |
| F-001 | Comparison returned 6 countries instead of the 2 asked | `country_resolver.py` + `compute.country_comparison()` — returns exactly the resolved set, never a default benchmark list |
| F-026, F-027 | "Compare" mis-routed to a generic ranking function; ranking had no explicit period | `intent_agent.py` schema forces `country_comparison` vs `country_ranking` to be distinguished by whether specific countries/periods were named |
| F-006, F-015, F-024, F-025 | Frequency/granularity guessed per-question | `period_resolver.choose_granularity()` — fixed rule: honor explicit ask, else finest granularity that actually has data; never silently switches |
| F-030 | Follow-up questions lost context | `SessionState` persisted server-side per session, referenced when `is_followup=true` |
| F-007, F-018, F-032–034 | No "overview/count/list" capability, fell back to single-indicator lookup | `_macro_overview()`, `count_indicators_by_type()`, `get_sector_indicators()` — deterministic, real counts/lists |
| F-016 | No capabilities answer | `_capabilities_answer()` — real counts pulled from the DB, not a canned lie |
| (implicit, your instruction) | LLM could invent any number in its final phrasing | `compute/verifier.py` — regex-checks every number in the Composer's output against the facts payload; falls back to a template if anything doesn't match |
| F-002, F-017 (Oxford Economics) | External forecast implausible | **Out of scope** — this build is SCAI-data-only; if a Combined mode with an external provider is still wanted, it needs its own separate honesty/plausibility layer, not covered here |

## What's still open / needs your input

1. **Embedding model server.** Semantic indicator matching needs a real
   embedding model (e.g. `BAAI/bge-m3`, multilingual for EN/AR) served
   separately from the 72B chat model — see `EMBEDDING_BASE_URL` in
   `app/core/config.py`. I validated the matching *logic* (confidence
   thresholds, ambiguity detection, contradiction checks) with simulated
   embeddings, since this sandbox has no network access to call a real
   embedding endpoint — recommend running the exact F-014/F-012/F-003
   test cases in `validate_etl.py`-style against the real model once it's
   up, before considering this closed.
2. **Macro overview indicator list** (`HEADLINE_NAMES` in `graph_v2.py`) is
   a placeholder editorial list (Real GDP, Inflation, Trade Balance,
   Government Revenues) — confirm the actual set SCAI wants surfaced for
   F-007/F-018-style "how's the economy doing" questions.
3. **Rounding convention** (headline 1 decimal, evidence full precision) is
   implemented in the verifier's tolerance check but not yet in a display
   layer — if the frontend needs the *evidence rows* separately from the
   headline, that's a payload-shape decision to confirm.
4. **Arabic composer testing** — the Composer prompt supports `language: ar`
   but hasn't been tested against real Arabic phrasing edge cases
   (F-019 "ما هو أخر تحليل للتضخم" and others) — recommend a dedicated pass
   once the embedding model (which needs to handle Arabic queries against
   English catalog names, or vice versa) is confirmed working end-to-end.
5. **Chart significant-figures/unit display** (F-004) is a frontend
   rendering concern — the facts payload already carries `unit_en` from
   `indicator_details`, so the frontend has what it needs; no backend change
   should be required beyond making sure that field is passed through.

## Source attribution on every answer

Every answer gets a `Sources:` footer, and every fact in the API's
`facts_payload.citations` array carries the exact backing record — this is
appended by code in `app/core/graph_v2._finish()`, not something the
Composer LLM is merely asked to remember. The same reasoning as the numeric
verifier: an instruction the LLM might skip on some phrasing isn't a real
guarantee, so it's enforced structurally instead.

A citation identifies:
- **Indicator name** and **original data source** (e.g. `World Bank`,
  `Qatar Tourism` — from `indicator_details.data_source_en`)
- **Which table** the number came from — `published_data_points` (SCAI-vetted,
  curated) vs `indicator_values` (raw/working data not yet published) —
  labeled honestly as such in the footer, not hidden
- **The exact record id** (`published_data_point_id` / `value_id`) — included
  in `facts_payload.citations` for any consumer that wants to link back to
  the precise source row, even though the prose footer keeps it human-readable
  and doesn't show raw GUIDs
- **Period and country**

Example footer for a min/max query:
```
Sources:
• Real GDP — SCAI Approved/Published Data — original source: World Bank — Q3-2025
```

Multiple periods (e.g. a trend query) collapse into one readable line rather
than one bullet per data point:
```
Sources:
• Real GDP — SCAI Approved/Published Data — original source: World Bank — Q1-2025 to Q4-2025 (4 data points)
```

Tested: a min/max citation correctly points at the row that actually won the
comparison (Q3-2025/186.0, per F-008), not whichever row happened to be
retrieved last.

## Chart output

Charts are attached deterministically — no LLM decides chart type or
generates chart data. Two fixed rules:

1. **Chart-request detection** (`app/compute/chart_request.py`) — a keyword
   check ("chart", "graph", "plot", "visualize") plus an implicit rule:
   any `trend`-type question gets a chart even without the word "chart"
   appearing, matching the test sheet's "show me how GDP has been trending"
   expectation.
2. **Chart type** (`app/compute/chart_builder.py`) — a fixed lookup table
   from `computation_type` to chart type (`trend` → line, `country_comparison`/
   `country_ranking`/`period_comparison` → bar, `macro_overview` → bar). Not
   a model decision — it's a small, stable mapping.

Critically, **the chart is built from the exact same `facts`/`rows` already
used for the text answer and citations** — there's no second retrieval or
recomputation, so the chart can never show a different number than the
prose does. This directly fixes **F-004** (chart didn't match the
indicator's real unit/significant figures): `unit_en` and decimal places are
read from `indicator_details.unit_en`/`format` — the same metadata already
resolved for the text answer, not a separate guess.

The API response now has a top-level `chart` field (also inside
`facts_payload.chart`) with `chart_type`, `unit`, `decimal_places`, and the
data array — ready for the frontend to render without re-deriving anything.

Single-value answers (`latest_value`, `min_max`, `difference`, `growth_rate`)
don't produce a chart — a single number isn't chart-shaped — unless you want
these rendered as a labeled value card with a target line, which is a
frontend decision the payload already has enough data to support (`actual`,
`target`, `unit`).
