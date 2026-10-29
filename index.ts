// omp-warden: a configurable model-classifier gate for omp tool calls.
//
// omp's native approval system (modes, bash.patterns, restrictedFolders, destructive safety
// prompts, grant caching, headless-yolo) owns the deterministic rules. This plugin adds the one
// thing omp has no native equivalent for: scoring an otherwise-allowed tool call through a model.
//
// A single `tool_call` hook runs a two-stage decider on calls that are in scope for a rule:
//   stage 1  a native judge (the `judge` model role) scores the call's risk in [0, 1];
//   stage 2  a chat model (the `llm` model role) decides the band between the rule's thresholds.
// The hook only ever blocks (or lets the call through to native approval), so it composes with
// whatever approval mode omp is in. Run it under `approvalMode: yolo` for a permissive default.

import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent";
import { loadConfig, type Config, type Rule } from "./config.ts";
import {
  classify as realClassify,
  conversationText as realConversationText,
  decideWithLlm as realDecideWithLlm,
  resolveRoleModel as realResolveRoleModel,
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
  resolveRoleModel(ctx: DeciderContext, role: string): Model | undefined;
  classify(ctx: DeciderContext, model: Model, rule: Rule, state: Record<string, string>, timeoutMs: number): Promise<number | undefined>;
  decideWithLlm(ctx: DeciderContext, model: Model, subject: string, conversation: string | undefined, maxTokens: number, timeoutMs: number): Promise<LlmVerdict | undefined>;
  conversationText(ctx: DeciderContext): string | undefined;
}

export const realEngine: Engine = {
  resolveRoleModel: realResolveRoleModel,
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

  // Stage 1: the judge-role classifier.
  let risk: number | undefined;
  const judgeModel = engine.resolveRoleModel(ctx, config.judgeRole);
  if (judgeModel) {
    const state = buildState(engine, ctx, rule, subject);
    risk = await engine.classify(ctx, judgeModel, rule, state, config.timeoutMs);
  }

  if (risk !== undefined) {
    if (risk < rule.allowBelow) {
      return undefined;
    }
    if (risk > rule.blockAbove) {
      return riskBlock(rule, risk);
    }
    // A risk between the thresholds falls through to the LLM fallback.
  }

  // Stage 2: the chat-model fallback. Also reached when stage 1 could not answer.
  const llmModel = engine.resolveRoleModel(ctx, config.llmRole);
  if (llmModel) {
    const conversation = engine.conversationText(ctx);
    const verdict = await engine.decideWithLlm(ctx, llmModel, subject, conversation, config.maxTokens, config.timeoutMs);
    if (verdict) {
      if (verdict.allow) {
        return undefined;
      }
      return { block: true, reason: `${verdict.reason} ${ASK_LINE}`.trim() };
    }
  }

  // Neither stage produced an answer.
  return onNoAnswer(config, ctx);
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
