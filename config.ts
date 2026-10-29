// Configuration for omp-warden.
//
// The plugin's own config file is the source of truth. omp has no runtime API that hands a
// plugin the values declared in its `omp.settings` manifest, so those are read back here from
// the plugin lock file and project overrides, and overlaid on the file config for the handful
// of scalar settings the manifest exposes.

import { readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

export type NoAnswer = "block" | "ask" | "allow";

const NO_ANSWER_VALUES: NoAnswer[] = ["block", "ask", "allow"];

export interface Rule {
  // A label shown in logs and block reasons.
  name: string;
  // Tool-name patterns this rule applies to. "*" matches every tool; "prefix*" matches by prefix.
  tools: string[];
  // risk < allowBelow runs without asking.
  allowBelow: number;
  // risk > blockAbove is blocked.
  blockAbove: number;
  // A risk between the two thresholds goes to the LLM fallback.
  // The yes/no question the stage-1 classifier answers about the call.
  question: string;
  // Whether stage 1 also sees a transcript of the conversation, not just the command.
  includeContext: boolean;
}

export interface Config {
  // Starting state. A false value turns the gate off and lets native omp approval decide alone.
  enabled: boolean;
  // Print a line to chat for each call checked, including the classifier's score.
  debug: boolean;
  // Tools that are never scored (native omp approval still applies). Optional in the file; omit to
  // use the built-in default list.
  whitelist: string[];
  // The omp model role whose model scores stage 1 (a native judge, e.g. a local tiny model).
  // "" disables the classifier stage. Optional in the file (defaults to "judge").
  judgeRole: string;
  // The omp model role whose chat model decides stage 2; falls back to the "default" role when it
  // is unset in omp. "" disables the llm stage. Optional in the file (defaults to "smol").
  llmRole: string;
  // What to do when neither stage can answer (model down, timeout, not a judge).
  onNoAnswer: NoAnswer;
  // Per-model-call timeout, in milliseconds.
  timeoutMs: number;
  // Reply budget for the stage-2 chat model.
  maxTokens: number;
  // Rules are tried in order; the first whose `tools` match the call decides it.
  rules: Rule[];
}

const HOME = os.homedir();

export const AGENT_DIR = process.env.PI_CODING_AGENT_DIR ?? path.join(HOME, ".omp/agent");
export const CONFIG_FILE = path.join(AGENT_DIR, "omp-warden", "config.json");

const PLUGIN_NAME = "omp-warden";
const LOCK_FILE = path.join(HOME, ".omp/plugins/omp-plugins.lock.json");

export const DEFAULT_RULE: Rule = {
  name: "default",
  tools: ["*"],
  allowBelow: 0.2,
  blockAbove: 0.8,
  question:
    "Could this tool call exfiltrate data, change a production system, or destroy data that the user did not explicitly ask to change?",
  includeContext: true,
};

export const DEFAULTS: Config = {
  enabled: true,
  debug: false,
  // Read-only and bookkeeping tools: searching, todos, questions to the user, subagents (whose
  // own calls come through this hook one by one), and thinking.
  whitelist: ["read", "grep", "glob", "find", "ast_grep", "lsp", "todo", "ask", "task", "think", "wait", "yield"],
  judgeRole: "judge",
  llmRole: "smol",
  onNoAnswer: "block",
  timeoutMs: 20000,
  maxTokens: 512,
  rules: [DEFAULT_RULE],
};

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isFraction(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

// A model-role config value: a role name, or null/"" to disable that stage. Returns "" for disabled.
function optionalRole(value: unknown): { value?: string; error?: string } {
  if (value === null || value === "") {
    return { value: "" };
  }
  if (typeof value === "string") {
    return { value };
  }
  return { error: "must be a role name string, or null to disable the stage" };
}

function validateRule(value: unknown, index: number): { rule?: Rule; error?: string } {
  if (!isPlainObject(value)) {
    return { error: `rule ${index} is not an object` };
  }
  const base = { ...DEFAULT_RULE };
  if (typeof value.name === "string") {
    base.name = value.name;
  }
  if ("tools" in value) {
    if (!isStringArray(value.tools)) {
      return { error: `rule ${index}: tools must be an array of strings` };
    }
    base.tools = value.tools;
  }
  if ("allowBelow" in value) {
    if (!isFraction(value.allowBelow)) {
      return { error: `rule ${index}: allowBelow must be a number in [0, 1]` };
    }
    base.allowBelow = value.allowBelow;
  }
  if ("blockAbove" in value) {
    if (!isFraction(value.blockAbove)) {
      return { error: `rule ${index}: blockAbove must be a number in [0, 1]` };
    }
    base.blockAbove = value.blockAbove;
  }
  if (base.allowBelow > base.blockAbove) {
    return { error: `rule ${index}: allowBelow (${base.allowBelow}) must not exceed blockAbove (${base.blockAbove})` };
  }
  if ("question" in value) {
    if (typeof value.question !== "string" || value.question.length === 0) {
      return { error: `rule ${index}: question must be a non-empty string` };
    }
    base.question = value.question;
  }
  if ("includeContext" in value) {
    if (typeof value.includeContext !== "boolean") {
      return { error: `rule ${index}: includeContext must be a boolean` };
    }
    base.includeContext = value.includeContext;
  }
  return { rule: base };
}

function validateRules(value: unknown): { rules?: Rule[]; error?: string } {
  if (!Array.isArray(value)) {
    return { error: "rules must be an array" };
  }
  if (value.length === 0) {
    return { error: "rules must not be empty" };
  }
  const rules: Rule[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const result = validateRule(value[index], index);
    if (result.error) {
      return { error: result.error };
    }
    rules.push(result.rule as Rule);
  }
  return { rules };
}

// Validates a parsed config object. Any error makes the whole config fall back to the defaults,
// so a single typo can never half-apply.
export function validateConfig(raw: unknown): { config: Config; error?: string } {
  if (!isPlainObject(raw)) {
    return { config: structuredClone(DEFAULTS), error: "config root is not an object" };
  }
  const config = structuredClone(DEFAULTS);
  const errors: string[] = [];

  if ("enabled" in raw) {
    if (typeof raw.enabled === "boolean") {
      config.enabled = raw.enabled;
    } else {
      errors.push("enabled must be a boolean");
    }
  }
  if ("whitelist" in raw) {
    if (isStringArray(raw.whitelist)) {
      config.whitelist = raw.whitelist;
    } else {
      errors.push("whitelist must be an array of strings");
    }
  }
  if ("debug" in raw) {
    if (typeof raw.debug === "boolean") {
      config.debug = raw.debug;
    } else {
      errors.push("debug must be a boolean");
    }
  }
  if ("judgeRole" in raw) {
    const role = optionalRole(raw.judgeRole);
    if (role.error) {
      errors.push(`judgeRole ${role.error}`);
    } else {
      config.judgeRole = role.value as string;
    }
  }
  if ("llmRole" in raw) {
    const role = optionalRole(raw.llmRole);
    if (role.error) {
      errors.push(`llmRole ${role.error}`);
    } else {
      config.llmRole = role.value as string;
    }
  }
  if ("onNoAnswer" in raw) {
    if (NO_ANSWER_VALUES.includes(raw.onNoAnswer as NoAnswer)) {
      config.onNoAnswer = raw.onNoAnswer as NoAnswer;
    } else {
      errors.push(`onNoAnswer must be one of: ${NO_ANSWER_VALUES.join(", ")}`);
    }
  }
  if ("timeoutMs" in raw) {
    if (typeof raw.timeoutMs === "number" && raw.timeoutMs > 0) {
      config.timeoutMs = raw.timeoutMs;
    } else {
      errors.push("timeoutMs must be a positive number");
    }
  }
  if ("maxTokens" in raw) {
    if (typeof raw.maxTokens === "number" && raw.maxTokens > 0) {
      config.maxTokens = raw.maxTokens;
    } else {
      errors.push("maxTokens must be a positive number");
    }
  }
  if ("rules" in raw) {
    const result = validateRules(raw.rules);
    if (result.error) {
      errors.push(result.error);
    } else {
      config.rules = result.rules as Rule[];
    }
  }

  if (errors.length > 0) {
    return { config: structuredClone(DEFAULTS), error: errors.join("; ") };
  }
  return { config };
}

function readJsonFile(file: string): { value?: unknown; error?: string } {
  let text: string;
  try {
    text = readFileSync(file, "utf8");
  } catch (e) {
    const err = e as NodeJS.ErrnoException;
    if (err.code === "ENOENT") {
      return {};
    }
    return { error: `cannot read ${file}: ${err.message}` };
  }
  try {
    return { value: JSON.parse(text) };
  } catch (e) {
    return { error: `cannot parse ${file}: ${(e as Error).message}` };
  }
}

// The settings this plugin declared in its manifest, as stored by omp for this plugin. The lock
// file's exact shape is not part of omp's public contract, so every known layout is checked and
// anything unexpected is ignored.
function pluginSettingsFrom(raw: unknown): Record<string, unknown> | undefined {
  if (!isPlainObject(raw)) {
    return undefined;
  }
  const candidates: unknown[] = [
    isPlainObject(raw.settings) ? raw.settings[PLUGIN_NAME] : undefined,
    isPlainObject(raw.config) && isPlainObject(raw.config.settings) ? raw.config.settings[PLUGIN_NAME] : undefined,
    isPlainObject(raw.plugins) && isPlainObject(raw.plugins[PLUGIN_NAME]) ? (raw.plugins[PLUGIN_NAME] as Record<string, unknown>).settings : undefined,
  ];
  for (const candidate of candidates) {
    if (isPlainObject(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

// Overlays the scalar settings the manifest exposes (the arrays stay file-only). Invalid or
// missing values are left untouched, so a stray lock-file entry can never break the config.
function applyScalarOverrides(config: Config, settings: Record<string, unknown> | undefined): void {
  if (!settings) {
    return;
  }
  if (typeof settings.enabled === "boolean") {
    config.enabled = settings.enabled;
  }
  if (typeof settings.debug === "boolean") {
    config.debug = settings.debug;
  }
  if (typeof settings.judgeRole === "string") {
    config.judgeRole = settings.judgeRole;
  } else if (settings.judgeRole === null) {
    config.judgeRole = "";
  }
  if (typeof settings.llmRole === "string") {
    config.llmRole = settings.llmRole;
  } else if (settings.llmRole === null) {
    config.llmRole = "";
  }
  if (NO_ANSWER_VALUES.includes(settings.onNoAnswer as NoAnswer)) {
    config.onNoAnswer = settings.onNoAnswer as NoAnswer;
  }
  if (typeof settings.timeoutMs === "number" && settings.timeoutMs > 0) {
    config.timeoutMs = settings.timeoutMs;
  }
  if (typeof settings.maxTokens === "number" && settings.maxTokens > 0) {
    config.maxTokens = settings.maxTokens;
  }
}

function projectOverrideFile(): string {
  return path.join(process.cwd(), ".omp", "plugin-overrides.json");
}

// Reads and validates the plugin config, then overlays any /settings values stored by omp.
// A config with any error is ignored as a whole and the defaults are used instead.
export function loadConfig(file: string = CONFIG_FILE): { config: Config; error?: string } {
  const read = readJsonFile(file);
  if (read.error) {
    return { config: structuredClone(DEFAULTS), error: read.error };
  }

  let config: Config;
  let error: string | undefined;
  if (read.value === undefined) {
    config = structuredClone(DEFAULTS);
  } else {
    const validated = validateConfig(read.value);
    config = validated.config;
    error = validated.error;
  }

  // The global lock file first, then project overrides, which take precedence.
  for (const overrideFile of [LOCK_FILE, projectOverrideFile()]) {
    const overrideRead = readJsonFile(overrideFile);
    if (!overrideRead.error) {
      applyScalarOverrides(config, pluginSettingsFrom(overrideRead.value));
    }
  }

  return { config, error };
}
