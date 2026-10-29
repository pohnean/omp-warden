// omp-warden: a configurable model-classifier gate for omp tool calls.
//
// omp's native approval system (modes, bash.patterns, restrictedFolders, destructive safety
// prompts, grant caching, headless-yolo) owns the deterministic rules. This plugin adds the one
// thing omp has no native equivalent for: scoring an otherwise-allowed tool call through a model.
//
// A single `tool_call` hook runs up to two stages on calls that are in scope for a rule:
//   stage 1  a native judge (the `judge` model role) scores the call's risk in [0, 1];
//   stage 2  a chat model (the `smol` role, falling back to `default`) decides the band between
//            the rule's thresholds, or the whole call when the classifier is off.
// Each stage is optional: set its role to "" to run classifier-only, llm-only, or neither. The
// hook only ever blocks (or lets the call through to native approval), so it composes with
// whatever approval mode omp is in. Run it under `approvalMode: yolo` for a permissive default.

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { loadConfig, type Config, type Rule } from "./config.ts";
import {
  classify as realClassify,
  conversationText as realConversationText,
  decideWithLlm as realDecideWithLlm,
  resolveRole as realResolveRole,
  type DeciderContext,
  type LlmVerdict,
  type Model,
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

// The decider functions grouped behind an interface so tests can supply fakes without reaching
// for pi-ai or a live model.
export interface Engine {
  resolveRole(ctx: DeciderContext, roles: string[]): Model | undefined;
  classify(ctx: DeciderContext, model: Model, rule: Rule, state: Record<string, string>, timeoutMs: number): Promise<number | undefined>;
  decideWithLlm(ctx: DeciderContext, model: Model, subject: string, conversation: string | undefined, maxTokens: number, timeoutMs: number): Promise<LlmVerdict | undefined>;
  conversationText(ctx: DeciderContext): string | undefined;
}

export const realEngine: Engine = {
  resolveRole: realResolveRole,
  classify: realClassify,
  decideWithLlm: realDecideWithLlm,
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

export function pickRule(rules: Rule[], tool: string): Rule | undefined {
  for (const rule of rules) {
    if (matchesTool(rule, tool)) {
      return rule;
    }
  }
  return undefined;
}

// A short description of the call for the models: the shell command for bash-like tools, or the
// tool name with its raw input otherwise.
export function describe(tool: string, input: Record<string, unknown>): string {
  if (typeof input.command === "string" && input.command.length > 0) {
    return input.command;
  }
  return `${tool} ${JSON.stringify(input ?? {})}`;
}

function clip(text: string, max = 200): string {
  if (text.length > max) {
    return `${text.slice(0, max)}…`;
  }
  return text;
}

function modelLabel(model: Model): string {
  const provider = typeof model.provider === "string" ? model.provider : "";
  const id = typeof model.id === "string" ? model.id : "";
  if (provider && id) {
    return `${provider}/${id}`;
  }
  if (id) {
    return id;
  }
  return "session model";
}

// The stage-2 role chain: the configured role, then "default" as a fallback.
function llmRoleChain(config: Config): string[] {
  if (config.llmRole === "default") {
    return ["default"];
  }
  return [config.llmRole, "default"];
}

// Prints a debug line to chat (when `debug` is on), via the UI notifier when the host exposes one,
// else to the plugin log.
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

function buildState(engine: Engine, ctx: DeciderContext, rule: Rule, subject: string): Record<string, string> {
  const state: Record<string, string> = { command: subject };
  if (rule.includeContext) {
    const conversation = engine.conversationText(ctx);
    if (conversation) {
      state.conversation = conversation;
    }
  }
  return state;
}

function riskBlock(rule: Rule, risk: number): BlockResult {
  const detail = `omp-warden judged this risky (score ${risk.toFixed(2)}, rule "${rule.name}" blocks above ${rule.blockAbove}).`;
  return { block: true, reason: `${detail} ${ASK_LINE}` };
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

  const rule = pickRule(config.rules, tool);
  if (!rule) {
    return undefined;
  }

  const input = (event.input ?? {}) as Record<string, unknown>;
  const subject = describe(tool, input);
  emitDebug(config, ctx, `checking ${tool} [rule "${rule.name}"]: ${clip(subject)}`);

  const useJudge = config.judgeRole !== "";
  const useLlm = config.llmRole !== "";
  if (!useJudge && !useLlm) {
    emitDebug(config, ctx, "no decider configured (both stages off) — passing through to native approval");
    return undefined;
  }

  // Stage 1: the judge-role classifier.
  let risk: number | undefined;
  if (useJudge) {
    const judgeModel = engine.resolveRole(ctx, [config.judgeRole]);
    if (judgeModel) {
      const state = buildState(engine, ctx, rule, subject);
      risk = await engine.classify(ctx, judgeModel, rule, state, config.timeoutMs);
      emitDebug(config, ctx, `stage 1 classifier (${config.judgeRole} → ${modelLabel(judgeModel)}): ${risk === undefined ? "no score" : `risk=${risk.toFixed(3)}`}`);
    } else {
      emitDebug(config, ctx, `stage 1 classifier: no model for role "${config.judgeRole}"`);
    }
  }

  if (risk !== undefined) {
    if (risk < rule.allowBelow) {
      emitDebug(config, ctx, `allow — risk ${risk.toFixed(3)} < allowBelow ${rule.allowBelow}`);
      return undefined;
    }
    if (risk > rule.blockAbove) {
      emitDebug(config, ctx, `block — risk ${risk.toFixed(3)} > blockAbove ${rule.blockAbove}`);
      return riskBlock(rule, risk);
    }
    if (!useLlm) {
      // Classifier-only: a risk that is not clearly over the block threshold runs.
      emitDebug(config, ctx, `allow — classifier-only, risk ${risk.toFixed(3)} within ${rule.allowBelow}..${rule.blockAbove}`);
      return undefined;
    }
    emitDebug(config, ctx, `escalating to stage 2 — risk ${risk.toFixed(3)} within ${rule.allowBelow}..${rule.blockAbove}`);
  } else if (useJudge && !useLlm) {
    // The classifier was meant to decide but could not score, and there is no fallback.
    emitDebug(config, ctx, "classifier-only and no score — applying onNoAnswer");
    return onNoAnswer(config, ctx);
  }

  // Stage 2: the chat-model fallback, or the only stage when the classifier is off.
  if (useLlm) {
    const llmModel = engine.resolveRole(ctx, llmRoleChain(config));
    if (llmModel) {
      const conversation = engine.conversationText(ctx);
      const verdict = await engine.decideWithLlm(ctx, llmModel, subject, conversation, config.maxTokens, config.timeoutMs);
      if (verdict) {
        emitDebug(config, ctx, `stage 2 llm (${modelLabel(llmModel)}): ${verdict.allow ? "allow" : `block — ${verdict.reason}`}`);
        if (verdict.allow) {
          return undefined;
        }
        return { block: true, reason: `${verdict.reason} ${ASK_LINE}`.trim() };
      }
      emitDebug(config, ctx, `stage 2 llm (${modelLabel(llmModel)}): no answer`);
    } else {
      emitDebug(config, ctx, `stage 2 llm: no model for role(s) ${llmRoleChain(config).join(", ")}`);
    }
    return onNoAnswer(config, ctx);
  }

  // Unreachable in practice (classifier-only paths all return above); allow as a safe default.
  return undefined;
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
