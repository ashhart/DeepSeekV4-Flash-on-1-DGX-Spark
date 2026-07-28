# DeepSeek V4 Flash on a DGX Spark (GB10) — Setup Recipe

Running DeepSeek V4 Flash locally on an NVIDIA **GB10 / DGX Spark** (128 GB unified
memory, ~121 GiB usable). The 81 GB model + speculative stack fits with room for a
**full 256K context** for a single user. OpenAI-compatible HTTP server, thinking +
tool calls, speculative decode — no llama.cpp, no vLLM.

---

## 1. Engine — DwarfStar (`ds4`)

DeepSeek V4 runs on **DwarfStar**, a self-contained native engine for DS4.

- Upstream (Metal-first, single-user CLI/agent): https://github.com/antirez/ds4
- CUDA / batched-serving fork we run on the Spark: https://github.com/Entrpi/ds4  (v0.4.x, reference machine = DGX Spark / GB10)

Build on the Spark:
```bash
git clone https://github.com/Entrpi/ds4
cd ds4
make cuda-spark        # Linux CUDA build tuned for DGX Spark / GB10 (sm_121)
```
Produces `./ds4-server` (HTTP server) and `./ds4` (CLI).
Other targets: `make cuda-generic` (other CUDA GPUs), `make` (macOS Metal), `make cpu` (diagnostics).

> **If you'll drive it with a thinking-mode coding agent (opencode, etc.), apply the
> patch in [Appendix A](#appendix-a--patch-thinking-mode-tool-call-salvage) *before*
> building** — it fixes agents stalling when the model emits a tool call without
> closing `</think>`.

---

## 2. Models (GGUF)

From `antirez/deepseek-v4-gguf` on Hugging Face, via the bundled downloader:
```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE      # <-- put your own HF token here
export DS4_GGUF_DIR=~/gguf
./download_model.sh q2-imatrix          # base model, ~81 GB (recommended for 96/128 GB boxes)
./download_model.sh mtp                 # MTP speculative draft head, ~3.5 GB
```

Files in `~/gguf`:

| file | size | role |
|---|---|---|
| `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` | 81 GB | base model |
| `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` | 3.6 GB | MTP draft (speculative) |
| `DSpark-drafter-Q2K-Q8.gguf` | 6.5 GB | *optional* block-drafter (deeper speculation) |

The **DSpark drafter is optional** — if present beside the model it's auto-attached for
deeper speculative decode; without it you still get MTP-2 speculation. Start with just
base + MTP if you don't have it.

---

## 3. Run it (the tuned command)

Single-user box, full 256K context:
```bash
export DS4_SERVER_COALESCE_MAX=2        # one stream: minimum KV banks -> large per-bank budget
./ds4-server \
  -m ~/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp ~/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --dspark ~/gguf/DSpark-drafter-Q2K-Q8.gguf \
  --host 0.0.0.0 --port 8000 \
  -c 262144
```
(There's also a one-command launcher `ds4-serve` that auto-detects the three GGUFs in
`~/gguf`, so it reduces to: `DS4_SERVER_COALESCE_MAX=2 ds4-serve -c 262144 --host 0.0.0.0 --port 8000`.)

### Why these settings (what we learned tuning it on the Spark)
- **`-c 262144` (full 256K):** on a 128 GB Spark, ONE stream fits the full window. Measured KV budget 7.6 GiB vs 4.4 GiB needed at 256K (~1.75× margin); ~110/121 GiB used at max.
- **`DS4_SERVER_COALESCE_MAX=2`** — the key knob. Pins the engine to the minimum number of KV banks so a single conversation gets a large per-bank budget. The default (32) splits the budget too thin on a Spark and starts **rejecting big prompts** ("cont admit rejected on comp-cache budget"). For one user, set this to 2.
- **Leave `DS4_SESSION_LAZY_GRAPH` unset** (default = lazy). Do NOT set it to `0` — the eager session graph wastes 2–4 GiB the serving path never uses.
- Scaling: `128K → 11.3 GiB budget`, `192K → 9.4`, `256K → 7.6`. Two *simultaneously-full* 256K streams don't fit at default headroom — for that add `DS4_BATCH_FIT_HEADROOM_MB=4096` and load-test.
- Cost of big context: prefill ≈ 600 tok/s → ~3 min for a 100K-token prompt, ~6–7 min for a full 256K. **Normal chats are unaffected** — this only bites on genuinely huge prompts.

---

## 4. Client settings

Point your client (OpenWebUI, opencode, etc.) at `http://<box>:8000/v1`, model id
`deepseek-v4-flash`, and set the model's **context window to `262144`** to match the server.

Give the client a **generous request timeout** (10+ min). Long thinking-mode turns at
big context can exceed a 2-minute default, and if the client auto-retries on timeout you
get a retry storm that pins the box. Also don't let an agent conversation grow to ~150K
tokens — at that depth every thinking turn is minutes; start fresh / compact context.

---

## 5. (Optional) Hot-swap behind one endpoint — llama-swap + OpenWebUI

Wrapper `run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
PORT="${1:?usage: run.sh <port>}"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/sbsa-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DS4_SERVER_COALESCE_MAX=2
exec /path/to/ds4/ds4-server --host 127.0.0.1 --port "$PORT" \
  -m ~/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --mtp ~/gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
  --dspark ~/gguf/DSpark-drafter-Q2K-Q8.gguf \
  -c 262144
```
llama-swap config entry:
```yaml
models:
  "deepseek-v4-flash":
    checkEndpoint: /v1/models
    cmd: /path/to/run.sh ${PORT}
    cmdStop: pkill -x ds4-server
    proxy: http://localhost:${PORT}
```
Then point OpenWebUI's OpenAI endpoint at the llama-swap listen address.

---

## Notes
- **Speculative decode** (MTP + DSpark) is on by default; live acceptance ~75–95%, ~3–4.5 tokens per model forward step — that's what keeps decode usable (~12–34 tok/s). Downgrade with `--no-dspark` (MTP only) or `--no-spec` (plain).
- **Thinking mode** is on by default (better quality). If you drive it with a coding agent that makes tool calls, DeepSeek occasionally emits a tool call *without closing* its `</think>`; the DSML then leaks as text and the agent stalls. The fix is in [Appendix A](#appendix-a--patch-thinking-mode-tool-call-salvage) — apply it before building. (Quick workaround without the patch: run the agent in non-thinking mode.)
- `--trace <file>` logs prompts / cache decisions / output for debugging.
- `./ds4-server --help` lists every flag (disk-KV offload, sampling, distributed, etc.).

---

## Appendix A — Patch: thinking-mode tool-call salvage

**Problem it fixes.** DeepSeek V4 in thinking mode + tools sometimes ends its turn with a
real, complete tool call but *forgets to close `</think>` first*. The engine (correctly)
won't execute a tool call from inside the model's scratchpad, so it drops it and the raw
`<｜DSML｜tool_calls>…` leaks into the assistant text — your coding agent gets no tool
call to run and **stalls**. The patch salvages exactly the safe case: if the generation
*ends* with a complete, well-formed tool-call block, it synthesizes the missing `</think>`
and re-parses, so the call executes and the preceding text becomes reasoning. A tool call
merely quoted *mid*-reasoning (more text after it) is still ignored.

**Apply it** (from the repo root, before building):
```bash
# save the diff below as thinking-fix.patch, then:
git apply thinking-fix.patch          # or: patch -p1 < thinking-fix.patch
make cuda-spark                       # rebuild the server with the fix
```

**Validate** (optional but recommended):
```bash
make ds4_test && ./ds4_test --server  # expect: "server: OK" / "ds4 tests: ok", exit 0
```
When the salvage fires at runtime you'll see this in the log, and the tool executes:
```
ds4-server: thinking left open; salvaging trailing tool call as the assistant action
```

**The patch** (against Entrpi/ds4 v0.4.x; if line numbers have drifted, the change is just
a new `dsml_trailing_tool_block()` helper plus a branch in the `!think_end` path of
`parse_generated_message_ex` — apply by hand from the hunks below):

```diff
diff --git a/ds4_server.c b/ds4_server.c
--- a/ds4_server.c
+++ b/ds4_server.c
@@ -4494,6 +4494,34 @@ static void split_reasoning_content(const char *text, size_t n, char **content_o
     free(s);
 }
 
+/* If `text` ends (ignoring trailing whitespace) with a complete tool_calls
+ * block, return a pointer to that block's opening marker; otherwise NULL.
+ *
+ * Used to salvage the common thinking-mode failure where the model finishes its
+ * turn with a real, complete tool call but forgets to close </think> first.
+ * Conservative on purpose: the block's close marker must be the last
+ * non-whitespace of the generation, so a tool call merely quoted or drafted
+ * mid-reasoning (with more reasoning after it) is NOT treated as an action. */
+static const char *dsml_trailing_tool_block(const char *text) {
+    if (!text) return NULL;
+    const char *const styles[][2] = {
+        { DS4_TOOL_CALLS_START,       DS4_TOOL_CALLS_END },
+        { DS4_TOOL_CALLS_START_SHORT, DS4_TOOL_CALLS_END_SHORT },
+        { "<tool_calls>",             "</tool_calls>" },
+    };
+    for (size_t i = 0; i < sizeof(styles) / sizeof(styles[0]); i++) {
+        const char *end = find_last_substr(text, styles[i][1]);
+        if (!end) continue;
+        if (*skip_ascii_ws(end + strlen(styles[i][1])) != '\0') continue;
+        const char *start = strstr(text, styles[i][0]);
+        if (!start || start >= end) continue;
+        for (const char *scan = start; (scan = strstr(scan + 1, styles[i][0])) && scan < end; )
+            start = scan;
+        return start;
+    }
+    return NULL;
+}
+
 static bool parse_generated_message_ex(const char *text, bool require_thinking_closed,
                                        char **content_out, char **reasoning_out,
                                        tool_calls *calls) {
@@ -4510,7 +4538,29 @@ static bool parse_generated_message_ex(const char *text, bool require_thinking_c
     if (require_thinking_closed) {
         const char *think_end = find_last_substr(text, "</think>");
         if (!think_end) {
-            /* Model did not close thinking, ignore any DSML in reasoning */
+            /* Model left <think> open.  DSML inside reasoning is normally not
+             * executable (a quotation, a draft, or a protocol explanation).
+             * But the most common real failure is the model ending its turn
+             * with a genuine, complete tool call yet forgetting to close
+             * </think> first -- dropping it stalls the client with no action.
+             * Salvage that narrow case: a well-formed tool_calls block whose
+             * close marker is the last non-whitespace of the generation.  We
+             * synthesize the missing </think> before the block and re-parse so
+             * the normal closed-think path splits reasoning/content/tool_calls.
+             * A call quoted mid-reasoning (text after it) is not trailing and
+             * stays ignored below. */
+            const char *blk = dsml_trailing_tool_block(text);
+            if (blk) {
+                fprintf(stderr, "ds4-server: thinking left open; salvaging trailing tool call as the assistant action\n");
+                buf synth = {0};
+                buf_append(&synth, text, (size_t)(blk - text));
+                buf_puts(&synth, "</think>\n\n");
+                buf_puts(&synth, blk);
+                bool ok = parse_generated_message_ex(synth.ptr ? synth.ptr : "</think>",
+                                                     true, content_out, reasoning_out, calls);
+                buf_free(&synth);
+                return ok;
+            }
             fprintf(stderr, "ds4-server: thinking not closed, ignoring DSML in reasoning\n");
             split_reasoning_content(text, strlen(text), content_out, reasoning_out);
             return true;
```

**Optional — the unit tests** (adds a positive + negative case; append the two functions
next to the other `test_thinking_dsml_*` tests and register them in
`ds4_server_unit_tests_run()`):

```diff
@@ static void ds4_server_unit_tests_run(void) {
     test_thinking_dsml_is_not_executable_before_think_close();
     test_thinking_dsml_after_think_close_is_executable();
+    test_thinking_dsml_trailing_tool_call_salvaged_when_think_unclosed();
+    test_thinking_dsml_midreasoning_not_salvaged_when_think_unclosed();
```
```c
static void test_thinking_dsml_trailing_tool_call_salvaged_when_think_unclosed(void) {
    /* Model forgot to close </think> but ended with a real, complete tool call:
     * salvage and execute it; preceding text becomes reasoning. */
    const char *generated =
        "<think>I need to read the file first.\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"read\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/x" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END "\n";
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 1);
    TEST_ASSERT(calls.v[0].name && !strcmp(calls.v[0].name, "read"));
    TEST_ASSERT(strstr(calls.v[0].arguments, "\"path\": \"/tmp/x\"") != NULL);
    TEST_ASSERT(reasoning && strstr(reasoning, "I need to read the file first") != NULL);
    TEST_ASSERT(content && content[0] == '\0');
    free(content); free(reasoning); tool_calls_free(&calls);
}

static void test_thinking_dsml_midreasoning_not_salvaged_when_think_unclosed(void) {
    /* A tool call drafted mid-reasoning (more text after it, no </think>)
     * is not the final action -- must stay non-executable. */
    const char *generated =
        "<think>Maybe I would call:\n\n"
        DS4_TOOL_CALLS_START "\n"
        DS4_INVOKE_START " name=\"read\">\n"
        DS4_PARAM_START " name=\"path\" string=\"true\">/tmp/x" DS4_PARAM_END "\n"
        DS4_INVOKE_END "\n"
        DS4_TOOL_CALLS_END
        "\nbut let me reconsider first.";
    char *content = NULL, *reasoning = NULL;
    tool_calls calls = {0};
    TEST_ASSERT(parse_generated_message_ex(generated, true, &content, &reasoning, &calls));
    TEST_ASSERT(calls.len == 0);
    free(content); free(reasoning); tool_calls_free(&calls);
}
```

Covers both streaming and non-streaming (both funnel through the same parser). Reversible —
just revert the hunks and rebuild.
