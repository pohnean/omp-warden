// omp-warden: a configurable model-classifier gate for omp tool calls.
//
// omp's native approval system (modes, bash.patterns, restrictedFolders, destructive safety
// prompts, grant caching, headless-yolo) owns the deterministic rules. This plugin adds the one
// thing omp has no native equivalent for: scoring an otherwise-allowed tool call through a model.
//
// A single `tool_call` hook evaluates every rule whose `tools` match, in ONE classifier request:
//   stage 1  a native judge (the `judge` role) answers each rule's allow/ask/block question;
//            risk = 1 - P(allow), and P(block) is what can hard-block;
//   stage 2  for a rule whose risk lands in its unsure band [allowBelow, askAbove), a chat model
//            (the `llm` role) gives a 3-way allow/ask/block verdict for the whole call.
// The worst verdict across rules wins; the tripped rule's `description` + risk is the reason shown.
// The hook only ever blocks or asks (never force-allows), so it composes with omp's approval mode.
// Run it under `approvalMode: yolo` for a permissive default.

import { createHash } from "node:crypto";
import { appendFileSync, mkdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { AGENT_DIR, CONFIG_DIR, loadConfig, parseYaml, type Config, type Rule } from "./config.ts";
import {
  type RoleSettings,
  describeRole,
  type ClassifyResult,
  classify as realClassify,
  conversationText as realConversationText,
  escalate as realEscalate,
  resolveRole as realResolveRole,
  type DeciderContext,
  type Model,
  type RuleScore,
  type RuleVerdict,
} from "./decider.ts";

export interface ToolCallEvent {
  toolName: string;
  input?: Record<string, unknown>;
  toolCallId?: string;
}

export interface BlockResult {
  block: true;
  reason: string;
}

type Band = "allow" | "escalate" | "ask" | "block";
const BAND_RANK: Record<Band, number> = { allow: 0, escalate: 1, ask: 2, block: 3 };

// The decider functions grouped behind an interface so tests can supply fakes without reaching
// for pi-ai or a live model.
export interface Engine {
  resolveRole(ctx: DeciderContext, roles: string[]): Model | undefined;
  classify(ctx: DeciderContext, model: Model, rules: Rule[], state: string, timeoutMs: number): Promise<ClassifyResult>;
  escalate(ctx: DeciderContext, model: Model, rules: Rule[], state: string, maxTokens: number, timeoutMs: number): Promise<Record<string, RuleVerdict> | undefined>;
  conversationText(ctx: DeciderContext): string | undefined;
}

export const realEngine: Engine = {
  resolveRole: realResolveRole,
  classify: realClassify,
  escalate: realEscalate,
  conversationText: realConversationText,
};

const ASK_LINE = "If this is intended, ask the user to approve it.";

export function matchesTool(rule: Rule, tool: string): boolean {
  for (const pattern of rule.tools) {
    if (pattern === "*" || pattern === tool) {
      return true;
    }
    if (pattern.endsWith("*") && tool.startsWith(pattern.slice(0, -1))) {
      return true;
    }
  }
  return false;
}

export function matchingRules(rules: Rule[], tool: string): Rule[] {
  return rules.filter((rule) => matchesTool(rule, tool));
}

// A short description of the call for the models: the shell command for bash-like tools, or the
// tool name with its raw input otherwise.
export function describe(tool: string, input: Record<string, unknown>): string {
  if (typeof input.command === "string" && input.command.length > 0) {
    return input.command;
  }
  return `${tool} ${JSON.stringify(input ?? {})}`;
}

// The working directory of a call: the session's, or a bash call's own `cwd` resolved against it.
function callCwd(tool: string, input: Record<string, unknown>, ctx: DeciderContext): string {
  const base = ctx.cwd ?? process.cwd();
  if (tool === "bash" && typeof input.cwd === "string" && input.cwd !== "") {
    return path.resolve(base, input.cwd);
  }
  return base;
}

// A hash of every word in the command that is an existing file (e.g. `bash backup.sh`), so an
// edited script no longer matches a call allowed for the session. Empty when it names no file.
export function scriptHash(subject: string, cwd: string): string {
  const hash = createHash("sha256");
  let found = false;
  for (const word of new Set(subject.split(/[\s;|&()<>'"`]+/))) {
    if (word === "") {
      continue;
    }
    const file = path.resolve(cwd, word);
    try {
      if (!statSync(file).isFile()) {
        continue;
      }
      hash.update(file).update("\0").update(readFileSync(file)).update("\0");
      found = true;
    } catch {
      // Not a file.
    }
  }
  return found ? hash.digest("hex").slice(0, 16) : "";
}

// What "Allow for this session" remembers: this exact call — tool, folder, command text and the
// contents of any script it names. Any difference asks again.
export function sessionKey(tool: string, input: Record<string, unknown>, ctx: DeciderContext): string {
  const cwd = callCwd(tool, input, ctx);
  const subject = describe(tool, input);
  return [tool, cwd, subject, scriptHash(subject, cwd)].join("\0");
}

// The `state` string the judge and the llm both see: the conversation (when includeContext) plus
// the proposed tool call. Mirrors how the model was evaluated in the benchmark.
export function renderState(tool: string, subject: string, conversation: string | undefined): string {
  const head = conversation ? `Conversation so far:\n${conversation}\n\n` : "";
  return `${head}Proposed tool call (tool: ${tool}):\n${subject}`;
}

function clip(text: string, max = 200): string {
  if (text.length > max) {
    return `${text.slice(0, max)}…`;
  }
  return text;
}

// The band a rule's answer falls in: block from P(block), the rest from risk = 1 - P(allow). A
// missing score becomes "escalate" when an llm can re-check it, otherwise "allow" (don't stop a
// call on a rule we couldn't score).
function bandOf(rule: Rule, score: RuleScore | undefined, useLlm: boolean): Band {
  if (score === undefined) {
    return useLlm ? "escalate" : "allow";
  }
  if (rule.blockAbove !== undefined && score.pBlock !== undefined && score.pBlock >= rule.blockAbove) {
    return "block";
  }
  if (score.risk >= rule.askAbove) {
    return "ask";
  }
  if (score.risk >= rule.allowBelow) {
    return "escalate";
  }
  return "allow";
}

// One call's debug summary, shown as a single notification: the call, each rule's score (and the
// llm's verdict when it was asked), the outcome and the time taken.
interface Trace {
  started: number;
  head: string;
  parts: Map<string, string>;
  outcome?: string;
}

// 0.048 -> ".05"
const short = (n: number): string => n.toFixed(2).replace(/^0(?=\.)/, "");

// ".95 b.90" — P(block) only for a rule that can block.
function tracePart(rule: Rule, score: RuleScore | undefined): string {
  if (score === undefined) {
    return "?";
  }
  const block = rule.blockAbove !== undefined && score.pBlock !== undefined ? ` b${short(score.pBlock)}` : "";
  return `${short(score.risk)}${block}`;
}

function formatScore(score: RuleScore | undefined): string {
  if (score === undefined) {
    return "?";
  }
  const block = score.pBlock === undefined ? "" : ` P(block)=${score.pBlock.toFixed(3)}`;
  return `risk=${score.risk.toFixed(3)}${block}`;
}

export const DEBUG_LOG = path.join(CONFIG_DIR, "debug.log");

// Appends a debug line to DEBUG_LOG, the one place debug output can always be found (omp's
// notifications are invisible in headless runs and subagents). Never throws.
function appendDebugLog(line: string): void {
  try {
    mkdirSync(CONFIG_DIR, { recursive: true });
    appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${line}\n`);
  } catch {
    // Logging must never break a decision.
  }
}

// When `debug` is on: a detail line, written to DEBUG_LOG only.
function emitDebug(config: Config, ctx: DeciderContext, message: string): void {
  if (!config.debug) {
    return;
  }
  appendDebugLog(`[omp-warden] ${message}`);
}

// When `debug` is on: the call's one-line summary, to DEBUG_LOG and as an omp notification (stderr
// when the host has no notifier).
function emitSummary(config: Config, ctx: DeciderContext, trace: Trace): void {
  if (!config.debug || trace.head === "") {
    return;
  }
  const scores = [...trace.parts].map(([name, part]) => `${name} ${part}`);
  const line = ["[omp-warden]", trace.outcome ?? "allow", "·", [trace.head, ...scores, `${Date.now() - trace.started}ms`].join(" · ")].join(" ");
  appendDebugLog(line);
  try {
    if (typeof ctx.ui?.notify === "function") {
      ctx.ui.notify(line, "info");
      return;
    }
  } catch {
    // Fall through to stderr.
  }
  console.error(line);
}

// The stage-2 role chain: the configured role, then "default" as a fallback.
function llmRoleChain(config: Config): string[] {
  if (config.llmRole === "default") {
    return ["default"];
  }
  return [config.llmRole, "default"];
}

// A rule that tripped: what it decided, why, and the classifier's score when there is one.
interface Finding {
  rule: Rule;
  verdict: "ask" | "block";
  reason: string;
  score?: RuleScore;
  by: "classifier" | "llm";
}

// "risk 0.98, P(block) 0.10 · asks at 0.2, blocks at P(block) 0.8 · decided by the llm"
function scoreLine(finding: Finding): string {
  const rule = finding.rule;
  const parts: string[] = [];
  if (finding.score) {
    const block = finding.score.pBlock === undefined ? "" : `, P(block) ${finding.score.pBlock.toFixed(2)}`;
    parts.push(`risk ${finding.score.risk.toFixed(2)}${block}`);
  } else {
    parts.push("no classifier score");
  }
  parts.push(`asks at ${rule.askAbove}${rule.blockAbove === undefined ? "" : `, blocks at P(block) ${rule.blockAbove}`}`);
  if (finding.by === "llm") {
    parts.push("decided by the llm");
  }
  return parts.join(" · ");
}

async function confirmWithUser(ctx: DeciderContext, title: string, message: string): Promise<boolean> {
  if (!ctx.hasUI || typeof ctx.ui?.confirm !== "function") {
    return false;
  }
  try {
    return await ctx.ui.confirm(title, message);
  } catch {
    return false;
  }
}

// Acts on a finding: a block stops the call; an ask prompts the user (or stops the call when there is
// no UI — a subagent can't answer). The rule's description is the title; reason and score are shown.
const ALLOW_ONCE = "Allow once";
const ALLOW_SESSION = "Allow for this session";
const BLOCK = "Block";

type Choice = "once" | "session" | "block";

// The ask dialog: Allow once / Allow for this session / Block where omp has a select dialog, else a
// yes/no confirm (no session option). No UI, a cancel or an error is a block.
async function askUser(ctx: DeciderContext, title: string, message: string): Promise<Choice> {
  if (!ctx.hasUI) {
    return "block";
  }
  if (typeof ctx.ui?.select === "function") {
    try {
      const picked = await ctx.ui.select(`${title}\n\n${message}`, [ALLOW_ONCE, ALLOW_SESSION, BLOCK], { signal: ctx.signal });
      if (picked === ALLOW_ONCE) {
        return "once";
      }
      if (picked === ALLOW_SESSION) {
        return "session";
      }
      return "block";
    } catch {
      return "block";
    }
  }
  const approved = await confirmWithUser(ctx, title, `${message}\n\nRun it anyway?`);
  return approved ? "once" : "block";
}

async function act(config: Config, ctx: DeciderContext, finding: Finding, tool: string, subject: string, trace: Trace, onSession: () => void): Promise<BlockResult | undefined> {
  const label = finding.rule.description;
  const scores = scoreLine(finding);
  emitDebug(config, ctx, `${finding.verdict} — ${finding.rule.name}: ${finding.reason} (${scores})`);
  if (finding.verdict === "block") {
    trace.outcome = `block ${finding.rule.name}`;
    return { block: true, reason: `omp-warden blocked this call — ${label}: ${finding.reason} (${scores}). ${ASK_LINE}` };
  }
  const message = `${finding.reason}\n\n${tool}: ${clip(subject, 300)}`;
  const choice = await askUser(ctx, `omp-warden: ${label}`, message);
  if (choice === "session") {
    onSession();
  }
  const answered = choice === "block" ? (ctx.hasUI ? "declined" : "blocked (no UI)") : choice;
  trace.outcome = `ask ${finding.rule.name} → ${answered}`;
  if (choice !== "block") {
    return undefined;
  }
  return { block: true, reason: `omp-warden needs your approval — ${label}: ${finding.reason} (${scores}). ${ASK_LINE}` };
}

async function onNoAnswer(config: Config, ctx: DeciderContext, trace?: Trace): Promise<BlockResult | undefined> {
  if (config.onNoAnswer === "allow") {
    if (trace) {
      trace.outcome = "no answer → allow";
    }
    return undefined;
  }
  if (config.onNoAnswer === "ask") {
    const approved = await confirmWithUser(ctx, "omp-warden: could not evaluate this call", "No model answered (down or timed out).\n\nRun it anyway?");
    if (trace) {
      trace.outcome = `no answer → ask → ${approved ? "approved" : "blocked"}`;
    }
    if (approved) {
      return undefined;
    }
  } else if (trace) {
    trace.outcome = "no answer → block";
  }
  return {
    block: true,
    reason: "omp-warden could not evaluate this call (no model answer) and is set to block (onNoAnswer). This was not a policy decision; retrying is fine if the model becomes available.",
  };
}

// The core decision for one tool call. Returns a block result, or undefined to let the call pass
// on to omp's native approval.
// `session` holds the calls the user allowed for this session (see sessionKey); createHandler owns
// one per omp session.
export async function decide(config: Config, engine: Engine, event: ToolCallEvent, ctx: DeciderContext, session: Set<string> = new Set()): Promise<BlockResult | undefined> {
  const trace: Trace = { started: Date.now(), head: "", parts: new Map() };
  try {
    return await decideTraced(config, engine, event, ctx, trace, session);
  } finally {
    emitSummary(config, ctx, trace);
  }
}

async function decideTraced(config: Config, engine: Engine, event: ToolCallEvent, ctx: DeciderContext, trace: Trace, session: Set<string>): Promise<BlockResult | undefined> {
  if (!config.enabled) {
    return undefined;
  }

  const tool = event.toolName;
  if (config.whitelist.includes(tool)) {
    return undefined;
  }

  const matching = matchingRules(config.rules, tool);
  if (matching.length === 0) {
    return undefined;
  }

  const input = (event.input ?? {}) as Record<string, unknown>;
  const subject = describe(tool, input);
  const key = sessionKey(tool, input, ctx);
  const allowForSession = () => {
    session.add(key);
  };
  if (session.has(key)) {
    trace.head = `${tool}: ${clip(subject, 60)}`;
    trace.outcome = "allow (session)";
    return undefined;
  }
  const conversation = config.includeContext ? engine.conversationText(ctx) : undefined;
  const state = renderState(tool, subject, conversation);
  emitDebug(config, ctx, `checking ${tool} [${matching.map((r) => r.name).join(", ")}]: ${clip(subject)}`);
  trace.head = `${tool}: ${clip(subject, 60)}`;

  const useJudge = config.judgeRole !== "";
  const useLlm = config.llmRole !== "";
  if (!useJudge && !useLlm) {
    emitDebug(config, ctx, "no decider configured — passing through to native approval");
    trace.outcome = "no decider → native";
    return undefined;
  }

  // Stage 1: the classifier scores every matching rule in one request. Rules it is unsure about (or
  // all of them, when it can't answer or is off) go on to the llm.
  let scores: Record<string, RuleScore> | undefined;
  let unsure: Rule[] = matching;

  if (useJudge) {
    const judgeModel = engine.resolveRole(ctx, [config.judgeRole]);
    emitDebug(config, ctx, `stage 1 judge: ${describeRole(ctx, config.judgeRole, judgeModel)}`);
    let classifyError = "no model for the judge role";
    if (judgeModel) {
      const result = await engine.classify(ctx, judgeModel, matching, state, config.timeoutMs);
      scores = result.scores;
      classifyError = result.error ?? "no scores";
    }
    if (scores) {
      emitDebug(config, ctx, `stage 1: ${matching.map((r) => `${r.name} ${formatScore(scores?.[r.name])}`).join(", ")}`);
      for (const rule of matching) {
        trace.parts.set(rule.name, tracePart(rule, scores[rule.name]));
      }
      const banded = matching.map((rule) => ({ rule, score: scores?.[rule.name], band: bandOf(rule, scores?.[rule.name], useLlm) }));
      let worst: Band = "allow";
      for (const entry of banded) {
        if (BAND_RANK[entry.band] > BAND_RANK[worst]) {
          worst = entry.band;
        }
      }
      if (worst === "allow") {
        emitDebug(config, ctx, "allow — every rule below its allowBelow");
        return undefined;
      }
      if (worst === "block" || worst === "ask") {
        // The rule that decides: highest P(block) among blockers, else highest risk among askers.
        const candidates = banded.filter((entry) => entry.band === worst);
        const key = (entry: (typeof banded)[number]) => (worst === "block" ? (entry.score?.pBlock ?? 0) : (entry.score?.risk ?? 0));
        const top = candidates.reduce((best, entry) => (key(entry) > key(best) ? entry : best));
        const reason =
          worst === "block"
            ? `The classifier is confident this must not run (P(block) at or above ${top.rule.blockAbove}).`
            : `The classifier rates this risky enough to need your approval (risk at or above ${top.rule.askAbove}).`;
        return act(config, ctx, { rule: top.rule, verdict: worst, reason, score: top.score, by: "classifier" }, tool, subject, trace, allowForSession);
      }
      // worst === "escalate"
      unsure = banded.filter((entry) => entry.band === "escalate").map((entry) => entry.rule);
      if (!useLlm) {
        emitDebug(config, ctx, `allow — unsure (${unsure.map((r) => r.name).join(", ")}) and no llm stage`);
        trace.outcome = "allow (unsure, no llm)";
        return undefined;
      }
      emitDebug(config, ctx, `escalating to llm — unsure: ${unsure.map((r) => r.name).join(", ")}`);
    } else {
      emitDebug(config, ctx, `stage 1: no classifier answer — ${classifyError}`);
      trace.parts.set("judge", "no answer");
      if (!useLlm) {
        return onNoAnswer(config, ctx, trace);
      }
    }
  }

  // Stage 2: the chat model judges the call against each unsure rule's own question.
  const llmModel = engine.resolveRole(ctx, llmRoleChain(config));
  if (!llmModel) {
    emitDebug(config, ctx, `stage 2 llm: no model for role(s) ${llmRoleChain(config).join(", ")}`);
    trace.parts.set("llm", "no model");
    return onNoAnswer(config, ctx, trace);
  }
  const verdicts = await engine.escalate(ctx, llmModel, unsure, state, config.maxTokens, config.timeoutMs);
  if (!verdicts || Object.keys(verdicts).length === 0) {
    emitDebug(config, ctx, "stage 2 llm: no answer");
    trace.parts.set("llm", "no answer");
    return onNoAnswer(config, ctx, trace);
  }

  let decision: Finding | undefined;
  for (const rule of unsure) {
    const given = verdicts[rule.name];
    // A rule the llm skipped is treated as "ask" — it was unsure already, so a human confirms.
    let verdict = given?.verdict ?? "ask";
    const reason = given?.reason || "The llm gave no verdict for this rule.";
    emitDebug(config, ctx, `stage 2 llm: ${rule.name} ${verdict} — ${reason}`);
    const scored = trace.parts.get(rule.name);
    trace.parts.set(rule.name, scored === undefined ? `llm ${verdict}` : `${scored}→${verdict}`);
    // Only a rule with blockAbove may block; otherwise its block becomes an ask.
    if (verdict === "block" && rule.blockAbove === undefined) {
      verdict = "ask";
    }
    if (verdict === "allow") {
      continue;
    }
    if (decision === undefined || (verdict === "block" && decision.verdict === "ask")) {
      decision = { rule, verdict, reason, score: scores?.[rule.name], by: "llm" };
    }
  }
  if (decision === undefined) {
    trace.outcome = "allow (llm)";
    return undefined;
  }
  return act(config, ctx, decision, tool, subject, trace, allowForSession);
}

export function createHandler(config: Config, engine: Engine = realEngine) {
  // Calls the user allowed for this session. In memory only: gone when omp exits.
  const sessionAllowed = new Set<string>();
  return async (event: ToolCallEvent, ctx: DeciderContext): Promise<BlockResult | undefined> => {
    try {
      return await decide(config, engine, event, ctx, sessionAllowed);
    } catch (error) {
      // A throwing tool_call handler fails closed in omp. Log it and apply the no-answer policy
      // instead, so an internal error honours the user's onNoAnswer choice rather than silently
      // blocking everything.
      console.error(`[omp-warden] error evaluating ${event?.toolName}: ${(error as Error).message}`);
      return onNoAnswer(config, ctx);
    }
  };
}

const hasGetModelRole = (settings: unknown): settings is RoleSettings =>
  typeof (settings as RoleSettings | undefined)?.getModelRole === "function";

// Last resort: `modelRoles` read straight from omp's global config.yml, re-read on every call so a
// change made in /settings is picked up. Project-level role overrides are not seen this way.
const CONFIG_YML = path.join(AGENT_DIR, "config.yml");
const fileRoleSettings: RoleSettings = {
  source: CONFIG_YML,
  getModelRole(role: string): string | undefined {
    try {
      const parsed = parseYaml(readFileSync(CONFIG_YML, "utf8")) as { modelRoles?: Record<string, unknown> } | null;
      const spec = parsed?.modelRoles?.[role];
      return typeof spec === "string" && spec.length > 0 ? spec : undefined;
    } catch {
      return undefined;
    }
  },
};

// omp doesn't hand the tool_call hook its role settings (ctx.settings has no getModelRole), so find
// them elsewhere: omp's settings object on the plugin API, else config.yml.
function hostRoleSettings(pi: ExtensionAPI): RoleSettings {
  const api = pi as unknown as { pi?: { settings?: unknown }; settings?: unknown };
  for (const [settings, source] of [
    [api.pi?.settings, "omp settings (pi.pi.settings)"],
    [api.settings, "omp settings (pi.settings)"],
  ] as const) {
    if (hasGetModelRole(settings)) {
      return {
        source,
        getModelRole: (role: string) => (settings as RoleSettings).getModelRole?.(role),
      };
    }
  }
  return fileRoleSettings;
}

// The hook's ctx, with role settings filled in when omp left them out. Object.create keeps every
// other property and method of the original ctx.
function withRoleSettings(ctx: DeciderContext, fallback: RoleSettings): DeciderContext {
  if (hasGetModelRole(ctx.settings)) {
    return ctx;
  }
  return Object.create(ctx, { settings: { value: fallback } }) as DeciderContext;
}

export default function (pi: ExtensionAPI): void {
  const loaded = loadConfig();
  if (loaded.error) {
    console.error(`[omp-warden] config ignored, using defaults: ${loaded.error}`);
  }
  const handler = createHandler(loaded.config);
  const roles = hostRoleSettings(pi);
  pi.on("tool_call", (event, ctx) => handler(event as ToolCallEvent, withRoleSettings(ctx as unknown as DeciderContext, roles)));
}
