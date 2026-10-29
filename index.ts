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

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { loadConfig, type Config, type Rule } from "./config.ts";
import {
  classify as realClassify,
  conversationText as realConversationText,
  escalate as realEscalate,
  resolveRole as realResolveRole,
  type DeciderContext,
  type EscalationResult,
  type Model,
  type RuleScore,
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
  classify(ctx: DeciderContext, model: Model, rules: Rule[], state: string, timeoutMs: number): Promise<Record<string, RuleScore> | undefined>;
  escalate(ctx: DeciderContext, model: Model, state: string, maxTokens: number, timeoutMs: number): Promise<EscalationResult | undefined>;
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

function formatScore(score: RuleScore | undefined): string {
  if (score === undefined) {
    return "?";
  }
  const block = score.pBlock === undefined ? "" : ` P(block)=${score.pBlock.toFixed(3)}`;
  return `risk=${score.risk.toFixed(3)}${block}`;
}

function emitDebug(config: Config, ctx: DeciderContext, message: string): void {
  if (!config.debug) {
    return;
  }
  const line = `[omp-warden] ${message}`;
  try {
    if (typeof ctx.ui?.notify === "function") {
      ctx.ui.notify(line, "info");
      return;
    }
  } catch {
    // Fall through to the log.
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

function blockResult(reason: string): BlockResult {
  return { block: true, reason: `omp-warden blocked this call — ${reason}. ${ASK_LINE}` };
}

// An "ask": prompt the user when there is a UI, otherwise block (a subagent can't answer).
async function askUser(ctx: DeciderContext, reason: string): Promise<BlockResult | undefined> {
  if (ctx.hasUI && typeof ctx.ui?.confirm === "function") {
    let approved = false;
    try {
      approved = await ctx.ui.confirm(`omp-warden — ${reason}. Run it anyway?`);
    } catch {
      approved = false;
    }
    if (approved) {
      return undefined;
    }
  }
  return { block: true, reason: `omp-warden needs your approval — ${reason}. ${ASK_LINE}` };
}

async function onNoAnswer(config: Config, ctx: DeciderContext): Promise<BlockResult | undefined> {
  if (config.onNoAnswer === "allow") {
    return undefined;
  }
  if (config.onNoAnswer === "ask" && ctx.hasUI && typeof ctx.ui?.confirm === "function") {
    let approved = false;
    try {
      approved = await ctx.ui.confirm("omp-warden could not evaluate this tool call. Run it anyway?");
    } catch {
      approved = false;
    }
    if (approved) {
      return undefined;
    }
  }
  return {
    block: true,
    reason: "omp-warden could not evaluate this call (no model answer) and is set to block (onNoAnswer). This was not a policy decision; retrying is fine if the model becomes available.",
  };
}

// The core decision for one tool call. Returns a block result, or undefined to let the call pass
// on to omp's native approval.
export async function decide(config: Config, engine: Engine, event: ToolCallEvent, ctx: DeciderContext): Promise<BlockResult | undefined> {
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
  const conversation = config.includeContext ? engine.conversationText(ctx) : undefined;
  const state = renderState(tool, subject, conversation);
  emitDebug(config, ctx, `checking ${tool} [${matching.map((r) => r.name).join(", ")}]: ${clip(subject)}`);

  const useJudge = config.judgeRole !== "";
  const useLlm = config.llmRole !== "";
  if (!useJudge && !useLlm) {
    emitDebug(config, ctx, "no decider configured — passing through to native approval");
    return undefined;
  }

  // Stage 1: the classifier scores every matching rule in one request.
  let scores: Record<string, RuleScore> | undefined;
  let driver: { rule: Rule; risk: number; pBlock?: number } | undefined;
  let needLlm = !useJudge; // llm-only: escalate every in-scope call straight to the llm

  if (useJudge) {
    const judgeModel = engine.resolveRole(ctx, [config.judgeRole]);
    if (judgeModel) {
      scores = await engine.classify(ctx, judgeModel, matching, state, config.timeoutMs);
    }
    if (scores) {
      emitDebug(config, ctx, `stage 1: ${matching.map((r) => `${r.name} ${formatScore(scores?.[r.name])}`).join(", ")}`);
      let worst: Band = "allow";
      for (const rule of matching) {
        const score = scores[rule.name];
        const risk = score?.risk;
        const band = bandOf(rule, score, useLlm);
        if (band === "allow") {
          continue;
        }
        const better = BAND_RANK[band] > BAND_RANK[worst] || (BAND_RANK[band] === BAND_RANK[worst] && driver !== undefined && (risk ?? rule.askAbove) > driver.risk);
        if (BAND_RANK[band] > BAND_RANK[worst] || driver === undefined || better) {
          driver = { rule, risk: risk ?? rule.askAbove, pBlock: score?.pBlock };
        }
        if (BAND_RANK[band] > BAND_RANK[worst]) {
          worst = band;
        }
      }
      if (worst === "allow") {
        emitDebug(config, ctx, "allow — every rule below its allowBelow");
        return undefined;
      }
      if (worst === "block") {
        const reason = `${driver!.rule.description} (P(block) ${(driver!.pBlock ?? 0).toFixed(2)})`;
        emitDebug(config, ctx, `block — ${driver!.rule.name} P(block) ${(driver!.pBlock ?? 0).toFixed(3)} >= blockAbove`);
        return blockResult(reason);
      }
      if (worst === "ask") {
        const reason = `${driver!.rule.description} (risk ${driver!.risk.toFixed(2)})`;
        emitDebug(config, ctx, `ask — ${driver!.rule.name} risk ${driver!.risk.toFixed(3)} >= askAbove`);
        return askUser(ctx, reason);
      }
      // worst === "escalate"
      if (!useLlm) {
        emitDebug(config, ctx, `allow — unsure (${driver!.rule.name} risk ${driver!.risk.toFixed(3)}) and no llm stage`);
        return undefined;
      }
      needLlm = true;
      emitDebug(config, ctx, `escalating to llm — unsure (${driver!.rule.name} risk ${driver!.risk.toFixed(3)})`);
    } else {
      emitDebug(config, ctx, "stage 1: no classifier answer");
      if (!useLlm) {
        return onNoAnswer(config, ctx);
      }
      needLlm = true;
    }
  }

  if (!needLlm) {
    return undefined;
  }

  // Stage 2: the chat model gives a 3-way verdict for the whole call.
  const llmModel = engine.resolveRole(ctx, llmRoleChain(config));
  if (!llmModel) {
    emitDebug(config, ctx, `stage 2 llm: no model for role(s) ${llmRoleChain(config).join(", ")}`);
    return onNoAnswer(config, ctx);
  }
  const result = await engine.escalate(ctx, llmModel, state, config.maxTokens, config.timeoutMs);
  if (!result) {
    emitDebug(config, ctx, "stage 2 llm: no answer");
    return onNoAnswer(config, ctx);
  }
  emitDebug(config, ctx, `stage 2 llm: ${result.verdict}${result.reason ? ` — ${result.reason}` : ""}`);
  if (result.verdict === "allow") {
    return undefined;
  }
  const label = driver ? driver.rule.description : "flagged on review";
  const reason = result.reason ? `${label}: ${result.reason}` : label;
  return result.verdict === "block" ? blockResult(reason) : askUser(ctx, reason);
}

export function createHandler(config: Config, engine: Engine = realEngine) {
  return async (event: ToolCallEvent, ctx: DeciderContext): Promise<BlockResult | undefined> => {
    try {
      return await decide(config, engine, event, ctx);
    } catch (error) {
      // A throwing tool_call handler fails closed in omp. Log it and apply the no-answer policy
      // instead, so an internal error honours the user's onNoAnswer choice rather than silently
      // blocking everything.
      console.error(`[omp-warden] error evaluating ${event?.toolName}: ${(error as Error).message}`);
      return onNoAnswer(config, ctx);
    }
  };
}

export default function (pi: ExtensionAPI): void {
  const loaded = loadConfig();
  if (loaded.error) {
    console.error(`[omp-warden] config ignored, using defaults: ${loaded.error}`);
  }
  const handler = createHandler(loaded.config);
  pi.on("tool_call", (event, ctx) => handler(event as ToolCallEvent, ctx as unknown as DeciderContext));
}
