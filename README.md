# omp-warden

A small [oh-my-pi (omp)](https://github.com/can1357/oh-my-pi) plugin that scores each tool call
through a model and blocks the risky ones. It is built for a local-model setup: the deterministic
rules are left to omp's own approval system, and this plugin adds the one thing omp has no native
equivalent for — a model deciding whether an otherwise-allowed call should run.

## Philosophy

omp already does the deterministic work natively, so this plugin does **not** reimplement it:

| Concern | Handled by native omp |
| --- | --- |
| Permissive default | `tools.approvalMode: yolo` |
| Allow/deny/prompt rules over commands | `bash.patterns` |
| Destructive-command prompts | built-in bash safety overrides |
| Protected paths | `tools.restrictedFolders` |
| "Always allow" memory | `always-ask-cached` + `approval-cache.db` |
| Subagent / no-UI handling | native headless-yolo (a prompt auto-rejects) |

This plugin is purely the **classifier gate**. Its `tool_call` hook runs *before* native approval
and only ever **blocks or asks** (otherwise the call falls through to native approval), so it
composes with whatever mode omp is in. Run omp in `yolo` for the intended "permissive by default,
catch the dangerous stuff" behaviour.

## How a call is decided

For a tool call that isn't on the whitelist, **every rule whose `tools` match is scored in one
request**, and the worst verdict wins — the tripped rule's `description` is the reason shown.

1. **Stage 1 — classifier.** The model on the omp `judge` role answers each rule's question with a
   probability for allow, ask and (if the rule has it) block. A native judge; a local tiny model
   works. With `risk = 1 − P(allow)`, per rule:
   - `risk < allowBelow` → **run**.
   - `allowBelow ≤ risk < askAbove` → **unsure** → stage 2 (or run, if the llm stage is off).
   - `askAbove ≤ risk` → **ask** the user.
   - `P(block) ≥ blockAbove` (optional) → **block** outright.
2. **Stage 2 — llm.** When a rule is unsure, the model on the omp `llm` role (default `smol`,
   falling back to `default`) gives a 3-way `ALLOW` / `ASK` / `BLOCK` for the whole call — reading
   the full command/script and the conversation. It also runs when the classifier can't score.
3. If neither stage answers, `onNoAnswer` decides (`block` by default). Without a UI an `ask`
   blocks — a subagent can't be prompted.

An ask offers **Allow once / Allow for this session / Block**. "For this session" remembers the
exact call — tool, working directory, command text and the contents of any file the command names
(e.g. `bash backup.sh`) — and lets it through without a model call until omp exits; any difference
asks again. It is in memory only, and a block is never offered it.

Every rule has the same shape: a question and what allow / ask / block mean for it (see
[Rules](#rules)). Set `judgeRole`/`llmRole` to `null` to run classifier-only, llm-only, or (both
`null`) leave everything to native omp.

## Install

1. Install it as an omp plugin — e.g. `omp plugin install /path/to/omp-warden` (a local path is
   linked, so edits take effect). Run `npm install` in the plugin dir so its one dependency (`yaml`)
   is present. The host provides the peer module `@oh-my-pi/pi-coding-agent`; `@oh-my-pi/pi-ai` is
   imported lazily at decision time.
2. Recommended omp setup in `~/.omp/agent/config.yml`:

   ```yaml
   tools:
     approvalMode: yolo
     # The hard rules this plugin intentionally does NOT implement — define them natively:
     restrictedFolders:
       - ~/.omp/agent          # protect omp's own config/instructions
       - ~/secrets
     # bash.patterns can deny or prompt destructive/root commands outright, before any model runs.

   modelRoles:
     judge: local/lfm2-1.2b    # the stage-1 classifier (local native judge)
     # default: local/glm-5.3-flash   # the stage-2 fallback, if you don't want the session model

   retry:
     fallbackChains:
       judge: []               # keep the judge fully local
   ```

## Configuration

The source of truth is the plugin's own config file (omp has no runtime API to hand a plugin its
`/settings` values). It can be **YAML or JSON** — the config dir is searched in this order and the
first match wins:

```
$PI_CODING_AGENT_DIR/omp-warden/config.yaml   # default ~/.omp/agent/omp-warden/config.yaml
                                config.yml
                                config.json
```

Copy `config.example.yaml` (or `config.example.jsonc`) and keep only the keys you change. YAML is
handy for multi-line `question` strings. Every top-level key is optional — omit it for its default.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Turn the gate off and let native approval decide alone. |
| `debug` | `false` | One notification per checked call, one line per stage that ran: `classifier: ASK auto-mode: .60, data-exfiltration: .11 b.01 360ms` (`b` = P(block), only for rules with `blockAbove`) and, when it was asked, `llm: ALLOW data-exfiltration: <reason> 3800ms`. Times are the models' only. Step-by-step detail goes to `~/.omp/agent/omp-warden/debug.log` (also in headless runs and subagents). |
| `includeContext` | `true` | Send the conversation so already-authorized calls aren't re-asked. |
| `whitelist` | reads, search, todo, task, … | Tools never scored (native approval still applies). Omit for the built-in list. Writes and edits to omp's `local://` scratch (where plan mode keeps its plan) and `xd://propose` are never scored either. |
| `judgeRole` | `"judge"` | Stage-1 classifier model role. `null` disables the classifier stage. |
| `llmRole` | `"smol"` | Stage-2 model role; falls back to `default`. `null` disables the llm stage. |
| `onNoAnswer` | `"block"` | When the judge can't answer: `block` \| `ask` \| `allow`. |
| `timeoutMs` | `20000` | Per-request timeout. |
| `maxTokens` | `512` | Reply budget for the llm stage. |
| `rules[]` | one `auto-mode` rule | See [Rules](#rules). |

Set `judgeRole`/`llmRole` to `null` to run **classifier-only**, **llm-only**, or (both `null`)
leave every in-scope call to native approval. A config with any error is ignored as a whole (the
defaults are used), so a typo can't half-apply.

### `/settings`

The scalar keys (`enabled`, `debug`, `includeContext`, `judgeRole`, `llmRole`, `onNoAnswer`) are
declared in the plugin manifest, so they appear under `omp /settings → Plugins → omp-warden` and
`omp plugin config`. Because omp doesn't feed plugin settings to a running extension, the plugin
reads those values back from the plugin lock file (`~/.omp/plugins/omp-plugins.lock.json`) and
project `.omp/plugin-overrides.json`, and they override the config file. `whitelist` and `rules`
are file-only.

## Rules

Every rule whose `tools` match a call is asked together (one request); the **worst verdict wins**
and the tripped rule's `description` is shown. A call matching no rule is left to native approval.
Every rule has the same shape:

```ts
interface Rule {
  name: string;          // unique id; also the question key in the request
  description: string;   // shown to you when the rule asks or blocks
  tools: string[];       // "*", "prefix*", or exact tool names (default ["*"])
  question: string;      // required
  answers: {             // required: what each answer means for this question
    allow: string;
    ask: string;
    block?: string;      // optional; a rule without it can only ask
  };
  allowBelow: number;    // risk (= 1 − P(allow)) below this runs              (default 0.1)
  askAbove: number;      // from allowBelow up to this the llm decides; at/above it asks (default 0.2)
  blockAbove?: number;   // P(block) at/above this blocks; needs answers.block; omit = never blocks
}
```

```yaml
rules:
  - name: data-exfiltration
    description: Sending data to an untrusted destination
    tools: [bash, web_fetch, fetch, http]
    question: >
      Considering only whether data leaves the user's trusted systems: should this call run, need
      the user's approval, or be blocked? Only the USER's own messages grant permission.
    answers:
      allow: "No data leaves, or the user explicitly asked to send exactly this data there."
      ask: "Data may leave to a destination the user did not clearly approve."
      block: "Secrets, credentials or private data are sent to an untrusted or public destination."
    askAbove: 0.3
    blockAbove: 0.8
```

With no `rules:` block you get one built-in `auto-mode` rule over every tool (its full question and
answers are in `config.example.yaml`): it asks at risk 0.2 and never blocks. On the 130-case
benchmark this allow/ask/block form beat yes/no questions on approvals (12 vs 29 of 52 safe calls)
at equal protection.

Unknown rule keys, a missing `question` or `answers`, a `blockAbove` without a `block` answer, or
duplicate names make the config invalid (defaults are used, and the error is logged).

## Development

```
node test.mjs
```

Tests replace the decider's model calls with a fake engine, so they run with no omp host and no
pi-ai. Requires Node ≥ 22.18 / 23.6 (TypeScript types are stripped on import).

## License

MIT.
