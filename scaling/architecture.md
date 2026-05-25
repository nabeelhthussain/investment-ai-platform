# Scaling Architecture: Investment AI Platform

*Design document covering production scale, passive monitoring, LLM resilience, cost structure, security, and where AI should not be deployed.*

---

## 1. Scaling Ingestion to Hundreds or Thousands of Companies

### Current Architecture (Two-Company Demo)
The current implementation runs sequentially: fetch → chunk → analyze → output, in a single Python process. This is appropriate for a demo but does not scale.

### Production Architecture

**Pipeline Orchestration: Apache Airflow or Prefect**

Each company gets its own DAG (Directed Acyclic Graph) with three layers of scheduling:

- **Event-driven (real-time):** SEC EDGAR RSS feeds and Oslo Børs Newsweb webhooks trigger ingestion within minutes of a new filing. 8-K filings and Oslo regulatory announcements are highest-priority — they often contain material information that creates immediate trading relevance.
- **Periodic light refresh (nightly):** Scrape IR pages, check for press releases, update news sentiment. Lightweight — primarily diff-checks against prior content.
- **Full re-ingestion (quarterly):** After each earnings cycle, re-process all documents to update the corpus with new filings and refresh the retrieval index.

**Worker Architecture: Celery + Redis**

```
[Scheduler DAG]
      ↓
[Task Queue (Redis)]
      ↓
[Celery Workers × N]   ← scale horizontally by adding workers
  ├── fetcher worker   (I/O bound — many small jobs, high concurrency)
  ├── parser worker    (CPU bound — PDF extraction)
  ├── embedder worker  (API bound — rate-limited by embedding provider)
  └── analyzer worker  (LLM API bound — expensive, prioritized queue)
```

Each worker type scales independently. Fetcher workers can run 50+ concurrent jobs because they're I/O bound. Analyzer workers are throttled by LLM API rate limits and cost.

**Storage: Qdrant with Namespaced Collections**

At scale, the in-memory BM25 retriever in this demo must be replaced with Qdrant:
- One collection per company, named `{ticker}_{exchange}` (e.g. `SOC_NYSE`)
- Dense vectors: `jina-embeddings-v3` with late chunking (best recall on financial text)
- Sparse vectors: SPLADE or BM25-as-sparse alongside dense in the same collection
- Qdrant's hybrid search with RRF merge runs server-side — no client-side merge logic needed at scale

**Throughput estimate at 1,000 companies:**
- Full initial ingestion: ~4 hours at 15 concurrent workers, ~$1,500 in API costs
- Ongoing: ~200 new filings/day across the universe, ~$30–50/day to process

---

## 2. Passive Monitoring for Parked Ideas

### Design Principle
A "parked idea" is a company that was researched but not acted on. The analyst should be automatically notified of material changes without manually re-checking each company.

### Watchlist Table Schema

```sql
CREATE TABLE watchlist (
    id              UUID PRIMARY KEY,
    ticker          VARCHAR(20) NOT NULL,
    exchange        VARCHAR(10),
    analyst_owner   VARCHAR(100),
    parked_date     DATE,
    last_checked    TIMESTAMP,
    last_material_change TIMESTAMP,
    sensitivity     ENUM('HIGH', 'MEDIUM', 'LOW'),  -- alert threshold
    notification_channel VARCHAR(200),  -- Slack webhook or email
    prior_verdict   ENUM('PROCEED', 'STOP', 'NEEDS_MORE_INFO'),
    notes           TEXT
);
```

### Monitoring Loop (Nightly Celery Beat Job)

For each parked company:

1. **Fetch new documents** since `last_checked`. Lightweight — only fetch new filings, not full re-ingestion.

2. **Materiality classification.** A fast, cheap LLM call (Haiku/GPT-4o-mini) with a binary classifier prompt: *"Does this new filing or announcement represent a material change to the investment thesis? Answer YES or NO with one sentence of reasoning."* Categories that auto-trigger HIGH priority:
   - New 8-K with Item 1.01 (material agreement), 2.01 (acquisition), 5.02 (leadership change)
   - Profit warning or guidance revision
   - New litigation or regulatory action
   - Significant insider buy or sell

3. **Delta summary generation.** If the materiality classifier flags YES, generate a 2-3 paragraph "what changed" summary comparing new documents to the prior corpus. This uses the PRIMARY model (Claude Sonnet).

4. **Push notification.** Send to the analyst's configured channel (Slack, email, Teams) with: company name, change type, one-sentence summary, link to full delta report.

5. **Update `last_checked` and `last_material_change`** in the watchlist table.

### Alert Example (Slack)
```
🟡 MATERIAL CHANGE DETECTED: Sable Offshore (SOC)
Type: 8-K — Item 1.03 (Bankruptcy)
Summary: SOC filed Chapter 11 on [date]. This materially changes the 
prior NEEDS_MORE_INFO verdict. Immediate review recommended.
[View full delta report →]
```

### Sensitivity Levels
- **HIGH:** Alert on any new filing, including routine 10-Q
- **MEDIUM:** Alert on 8-Ks, earnings calls, guidance changes only (default)
- **LOW:** Alert only on 8-K items 1.01, 2.01, 5.02 (major events only)

---

## 3. Future-Proofing Against LLM API Changes

### The Core Risk
LLM providers change pricing, deprecate models, suffer outages, and introduce capability regressions. A system hard-coded to one provider is fragile.

### LiteLLM Router with Fallback Chains

All LLM calls in this system route through a single `router.py` abstraction. In production, this becomes a LiteLLM proxy with YAML-configured fallback chains:

```yaml
model_list:
  - model_name: primary
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
      
  - model_name: primary  
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
      
  - model_name: primary
    litellm_params:
      model: ollama/llama3.1:70b  # self-hosted fallback
      api_base: http://localhost:11434

router_settings:
  routing_strategy: fallback
  fallback_models: ["openai/gpt-4o", "ollama/llama3.1:70b"]
  num_retries: 3
```

If Anthropic's API returns a 529 (overloaded) or raises prices significantly, the system automatically routes to the fallback without code changes.

### Model Abstraction Principle
No agent file contains a model name. All model selection is in `config.py`:
- `PRIMARY_MODEL`: reasoning-heavy tasks (bear case, verdict, contradiction)
- `FAST_MODEL`: cheap extraction tasks (metadata, brief generation)
- Swapping models = editing two config variables

### Self-Hosted Fallback
For cost-sensitive operations (chunking metadata extraction, materiality classification), maintain the option to run `llama3.1:8b` or `mistral-7b` via Ollama on a small GPU instance. This provides:
- Zero marginal cost per call
- No rate limits
- Continued operation during API outages

The quality trade-off is acceptable for binary classifiers and metadata extraction, but NOT for the adversarial analysis agents where reasoning quality is the product.

### Versioned Prompt Library
Store all system prompts and agent prompts in version-controlled YAML files, not in Python strings. When a new model version changes behavior (common between major Claude versions), prompts can be updated and rolled back without touching agent logic.

---

## 4. Cost Structure

### Per-Company, Per-Full-Run

| Component | Notes | Estimated Cost |
|---|---|---|
| SEC EDGAR fetching | Free API | $0.00 |
| Web scraping | Compute only | ~$0.01 |
| PDF parsing | Compute only | ~$0.01 |
| Embedding (jina-v3 API) | ~500k tokens per company | ~$0.05 |
| LLM: bear case agent | ~8k tokens in + 3k out | ~$0.35 |
| LLM: contradiction agent | ~8k tokens in + 3k out | ~$0.35 |
| LLM: verdict agent | ~8k tokens in + 3k out | ~$0.35 |
| LLM: gap analysis | ~6k tokens in + 3k out | ~$0.28 |
| LLM: ingestion report summaries | Haiku model | ~$0.05 |
| LLM: analyst brief | Haiku model | ~$0.03 |
| Qdrant Cloud storage | Per GB/month at scale | ~$0.002/company/month |
| **Total per full run** | | **~$1.50–$2.50** |

### At Scale (1,000 companies)

| Scenario | Frequency | Annual Cost |
|---|---|---|
| Initial ingestion (1,000 companies) | Once | ~$2,000 |
| Quarterly full refresh | 4×/year | ~$8,000/year |
| Nightly passive monitoring (cheap classifier) | 365×/year | ~$1,800/year |
| Ad-hoc analyst-initiated re-runs | ~500/year | ~$1,000/year |
| Infrastructure (Qdrant Cloud, workers, Redis) | Monthly | ~$3,000/year |
| **Total annual at 1,000 companies** | | **~$16,000/year** |

This is approximately $16/company/year for continuous monitoring and quarterly deep refreshes — competitive with any commercial data provider.

### Cost Controls
- Use `claude-haiku-4-5-20251001` for all tasks that don't require deep reasoning (saves 20× vs Sonnet on eligible tasks)
- Cache retrieval results — many agent queries overlap; a cache hit costs $0
- Rate-limit the passive monitoring tier: don't run full dossier on every new 8-K, only on HIGH-priority material events
- Track cost per run in a `pipeline_runs` table; alert if any run exceeds 3× the expected cost

---

## 5. Security and Compliance

### Data Classification
| Data Type | Classification | Handling |
|---|---|---|
| Public SEC filings | Public | Standard storage |
| Public Oslo Børs filings | Public | Standard storage |
| AI-generated analyses | Confidential | Treat as internal research |
| Investment verdicts | Highly Confidential | Role-based access, audit log |
| API keys | Secret | Vault only, never logged |

### API Key Management
- All API keys in HashiCorp Vault or AWS Secrets Manager
- Keys rotated every 90 days automatically
- No keys in environment variables in production (only in Vault)
- Service accounts with minimal permissions (read-only SEC EDGAR access, etc.)

### Audit Logging
Every LLM call is logged with: timestamp, model used, prompt hash (not content), token counts, cost, output hash, requesting user/agent. This creates a non-repudiable audit trail for compliance review.

### Information Barriers
If the platform is used at a firm with multiple strategies (long/short, advisory), AI-generated analyses must respect the same information barriers as human research. Implement per-user company access lists — an analyst on the long book should not be able to query the AI's analysis of companies in the short book.

### Regulatory Considerations
- **EU AI Act (2024):** AI systems used to recommend financial investments may qualify as "high-risk." Document the human-in-the-loop requirement clearly — the system produces research inputs, not investment decisions.
- **MiFID II / FINRA:** AI-generated research may be subject to research unbundling and recordkeeping rules. Consult compliance before sharing outputs externally.
- **GDPR:** This pipeline processes company/issuer data, not personal data. Lower risk. However, if analyst names, email addresses, or personal trading data are stored alongside AI outputs, GDPR treatment is required.
- **Model documentation:** Maintain records of which LLM version produced each analysis, in case a specific output needs to be audited or challenged.

### Data Retention
- Raw source documents: retain indefinitely (public domain)
- AI-generated analyses: retain per firm's normal research retention policy (typically 7 years for regulated firms)
- Pipeline run logs: 2 years

---

## 6. Where NOT to Deploy AI

This is as important as the architecture itself.

### Do Not Automate Final Investment Decisions
The AI's verdict (PROCEED / STOP / NEEDS_MORE_INFO) is a structured recommendation to the analyst, not an autonomous decision. No position should be entered, exited, or sized based solely on AI output. A human analyst must review and sign off.

**Why:** LLMs can confidently produce incorrect factual claims (hallucination), miss non-public material information that an analyst would catch, and cannot incorporate relationship context, fund constraints, or portfolio-level considerations.

### Do Not Use AI for Client-Facing Research Notes Without Review
AI-generated analysis must not flow directly to clients or external parties. It must pass through compliance and human editorial review first.

### Do Not Rely on AI-Extracted Financials Without Verification
The Phase 4 financial model extraction is explicitly flagged as requiring human validation. LLMs can misread tables, confuse fiscal year conventions, and miss restatements. Any financial figure that flows into a model must be cross-checked against the source document.

### Do Not Deploy Without Human Review for Non-English Filings
For AKSO and other non-English filers: do not treat machine-translated content as equivalent to the original. Material nuances are frequently lost in translation. Norwegian-language filings at minimum need a human to verify the AI's interpretation of key passages.

### Do Not Use This System as a Substitute for Legal/Regulatory Due Diligence
The gap analysis agent flags legal proceedings and regulatory risks based on disclosed text. It cannot identify undisclosed risks, conduct independent legal research, or assess likelihood of regulatory outcomes. For any company with significant legal exposure, human legal review is required.

### Do Not Trust Confidence Scores Without Understanding Their Basis
The system outputs confidence levels (HIGH / MEDIUM / LOW) based on document coverage and internal model calibration. These are heuristics, not statistical confidence intervals. Low coverage + HIGH confidence is a red flag that should trigger analyst skepticism, not comfort.

---

## 7. What I Would Do Differently With More Time

### Architecture
- **True late chunking** with `jina-embeddings-v3` instead of the BM25 proxy — the quality improvement on financial text is meaningful
- **LangGraph for agent orchestration** — allows parallel agent execution (bear case + contradiction run simultaneously), checkpoint/resume on failure, and human-in-the-loop interrupts
- **Dedicated earnings call transcript pipeline** — Whisper ASR on conference call recordings + speaker diarization to identify CFO vs CEO statements separately

### Analysis Quality
- **Cross-company comparison** — for AKSO, compare order backlog trajectory against TechnipFMC, Subsea7, and Baker Hughes to provide sector context the current implementation lacks
- **FinBERT tone analysis** as a quantitative supplement to the contradiction detection — assign sentiment scores to each earnings call paragraph and plot the trend
- **Automated financial ratio extraction** — even without Phase 4 full model extraction, key ratios (net debt/EBITDA, revenue growth, margin trajectory) can be extracted with high reliability and should be in the dossier

### Infrastructure
- **Streaming output** — pipe agent outputs to a simple web UI as they generate, rather than batch-writing markdown files at the end
- **Evaluation harness** — a suite of test cases with known-correct answers for each agent, run automatically on every code change

### Known Weaknesses in Current Implementation
1. **AKSO PDFs may be inaccessible** — direct PDF URLs sometimes require authentication. The fallback is minimal web content, which reduces analysis quality significantly.
2. **No earnings call transcripts** — the most valuable source for tone analysis is the hardest to get programmatically without a paid data provider (Refinitiv, Bloomberg, Seeking Alpha Premium).
3. **BM25 retrieval quality** — adequate for a demo, meaningfully worse than dense embedding retrieval on semantic queries like "what does management think about energy transition strategy?"
4. **No deduplication across agents** — the same chunk may be retrieved by multiple agents, counting it multiple times in "chunks used" metrics.
5. **Rate limiting** — the tenacity retry logic handles transient errors but the pipeline will slow significantly if Anthropic API rate limits are hit during a full two-company run.
