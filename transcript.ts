// The context both stages read: the conversation, the call under review and the scripts it runs.
//
// It is built so text the agent wrote or read cannot pass as the user. Only the user's own messages
// grant permission, so:
//   - a user message is the user's typed words: an expanded skill body or an attached @file in front
//     of them is cut out;
//   - everything else (the agent's earlier calls, their output, script bodies) is JSON-quoted, so a
//     newline inside it can never start a line that reads as `[USER]`;
//   - the agent's own prose is never sent;
//   - when the budget runs out, tool lines go first and the user's first message (the task) is
//     always kept.

import { readFileSync, statSync } from "node:fs";
import path from "node:path";

export interface ContextLimits {
  // Budget for the conversation, in characters.
  maxChars: number;
  // One user message longer than this is clipped in the middle.
  userMessageChars: number;
  // One tool call's input or one tool output longer than this is clipped.
  toolChars: number;
  // Budget for the scripts the call runs.
  scriptChars: number;
  // Send the output of the agent's earlier tool calls (quoted) as well as the calls themselves.
  includeToolOutput: boolean;
}

export const DEFAULT_LIMITS: ContextLimits = {
  maxChars: 6000,
  userMessageChars: 2000,
  toolChars: 600,
  scriptChars: 8000,
  includeToolOutput: true,
};

export type Item =
  | { kind: "user"; text: string }
  | { kind: "call"; tool: string; input: unknown; outcome?: "ok" | "error" }
  | { kind: "output"; tool: string; text: string };

// "abc…(1234 chars)…xyz": keeps both ends, where commands and conclusions usually are.
export function clipMiddle(text: string, max: number): string {
  if (text.length <= max) {
    return text;
  }
  const keep = Math.max(0, max - 20);
  const head = Math.ceil(keep / 2);
  return `${text.slice(0, head)} …(${text.length} chars)… ${text.slice(text.length - (keep - head))}`;
}

// The user's own words in a user message: omp puts an expanded skill, or the files of `@file`, in
// front of them. Both cuts end at the last closing tag, so a skill body or a file can't close its
// block early and pass what follows as the user's words.
export function ownWords(text: string): string {
  const skill = /^<skill name="[^"]+" location="[^"]+">\n[\s\S]*\n<\/skill>(?:\n\n([\s\S]+))?$/.exec(text);
  if (skill) {
    return skill[1] ?? "";
  }
  const start = text.indexOf("<file name=");
  const end = text.lastIndexOf("</file>\n");
  if (start < 0 || end < start) {
    return text;
  }
  return text.slice(0, start) + text.slice(end + "</file>\n".length);
}

function textParts(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const pieces: string[] = [];
  for (const part of content) {
    if (typeof part === "object" && part !== null && (part as { type?: unknown }).type === "text") {
      const text = (part as { text?: unknown }).text;
      if (typeof text === "string") {
        pieces.push(text);
      }
    }
  }
  return pieces.join("\n");
}

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  typeof value === "object" && value !== null ? (value as Record<string, unknown>) : undefined;

// The conversation before the call under review, from omp's session branch. `pendingId` is the
// call's id: it, and any calls after it in the same message, haven't run and are left out.
export function transcriptItems(branch: unknown[], pendingId: string | undefined, includeToolOutput: boolean): Item[] {
  // Which earlier calls failed, by call id.
  const failed = new Map<string, boolean>();
  for (const entry of branch) {
    const message = asRecord(asRecord(entry)?.message);
    if (message?.role === "toolResult" && typeof message.toolCallId === "string") {
      failed.set(message.toolCallId, message.isError === true);
    }
  }

  const items: Item[] = [];
  for (const entry of branch) {
    const message = asRecord(asRecord(entry)?.message);
    if (message === undefined) {
      continue;
    }
    if (message.role === "user") {
      const text = ownWords(textParts(message.content)).trim();
      if (text !== "") {
        items.push({ kind: "user", text });
      }
      continue;
    }
    if (message.role === "assistant" && Array.isArray(message.content)) {
      for (const block of message.content) {
        const part = asRecord(block);
        if (part?.type !== "toolCall") {
          continue;
        }
        if (pendingId !== undefined && part.id === pendingId) {
          return items;
        }
        const id = typeof part.id === "string" ? part.id : undefined;
        const outcome = id !== undefined && failed.has(id) ? (failed.get(id) ? "error" : "ok") : undefined;
        items.push({ kind: "call", tool: String(part.name ?? ""), input: part.arguments, outcome });
      }
      continue;
    }
    if (message.role === "toolResult" && includeToolOutput) {
      const text = textParts(message.content).trim();
      if (text !== "") {
        items.push({ kind: "output", tool: String(message.toolName ?? ""), text });
      }
    }
  }
  return items;
}

// A tool name as written: plain names as is, anything odd quoted so it can't carry a line break.
const toolName = (name: string): string => (/^[\w.:-]+$/.test(name) ? name : JSON.stringify(name));

function line(item: Item, limits: ContextLimits): string {
  if (item.kind === "user") {
    return `[USER] ${clipMiddle(item.text, limits.userMessageChars)}`;
  }
  if (item.kind === "call") {
    const input = JSON.stringify(item.input ?? {}) ?? "{}";
    const outcome = item.outcome === undefined ? "" : ` → ${item.outcome}`;
    return `[TOOL CALL] ${toolName(item.tool)} ${JSON.stringify(clipMiddle(input, limits.toolChars))}${outcome}`;
  }
  return `[TOOL OUTPUT] ${toolName(item.tool)} ${JSON.stringify(clipMiddle(item.text, limits.toolChars))}`;
}

// The conversation as lines within the budget. Tool lines go first (oldest first), then user
// messages (oldest first) — except the user's first message, the task, which always stays.
export function renderTranscript(items: Item[], limits: ContextLimits): string | undefined {
  const lines = items.map((item) => ({ user: item.kind === "user", text: line(item, limits) }));
  const firstUser = lines.findIndex((entry) => entry.user);
  const keep = lines.map(() => true);
  let size = lines.reduce((total, entry) => total + entry.text.length + 1, 0);
  for (const wantUser of [false, true]) {
    for (let i = 0; i < lines.length && size > limits.maxChars; i++) {
      if (!keep[i] || lines[i].user !== wantUser || i === firstUser) {
        continue;
      }
      keep[i] = false;
      size -= lines[i].text.length + 1;
    }
  }
  const kept = lines.filter((_, i) => keep[i]).map((entry) => entry.text);
  if (kept.length === 0) {
    return undefined;
  }
  return kept.join("\n");
}

// Every word in the command that is an existing file under or relative to `cwd` (e.g. the
// `backup.sh` in `bash backup.sh`).
export function namedFiles(subject: string, cwd: string): string[] {
  const files: string[] = [];
  for (const word of new Set(subject.split(/[\s;|&()<>'"`]+/))) {
    if (word === "") {
      continue;
    }
    const file = path.resolve(cwd, word);
    try {
      if (statSync(file).isFile()) {
        files.push(file);
      }
    } catch {
      // Not a file.
    }
  }
  return files;
}

// The scripts the call names, quoted, for the models to judge with the call. Binary files are
// named but not included; the whole is clipped to the budget.
export function readScripts(subject: string, cwd: string, maxChars: number): string | undefined {
  const parts: string[] = [];
  for (const file of namedFiles(subject, cwd)) {
    const name = file.startsWith(cwd + path.sep) ? path.relative(cwd, file) : file;
    let body: string;
    try {
      body = readFileSync(file, "utf8");
    } catch (error) {
      parts.push(`[SCRIPT] ${JSON.stringify(name)} (cannot be read: ${(error as Error).message})`);
      continue;
    }
    if (body.includes("\0")) {
      parts.push(`[SCRIPT] ${JSON.stringify(name)} (binary, not shown)`);
      continue;
    }
    parts.push(`[SCRIPT] ${JSON.stringify(name)} ${JSON.stringify(body)}`);
  }
  if (parts.length === 0) {
    return undefined;
  }
  return clipMiddle(parts.join("\n"), maxChars);
}
