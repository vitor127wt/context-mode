/**
 * OpenCode / KiloCode TypeScript plugin entry point for context-mode.
 *
 * Provides five hooks (v1.0.107 — Mickey OC-1..OC-4 follow-up):
 *   - tool.execute.before  — Routing enforcement (deny/modify/passthrough)
 *   - tool.execute.after   — Session event capture + first-fire AGENTS.md scan (OC-4)
 *   - experimental.session.compacting — Compaction snapshot + budget-capped auto-injection (OC-3)
 *   - experimental.chat.system.transform — ROUTING_BLOCK + resume snapshot injection (OC-1)
 *   - chat.message         — User-prompt capture w/ CCv2 inline filter (OC-2) + AGENTS.md scan (OC-4)
 *
 * KiloCode loads this via: import("context-mode") → expects default export
 * with shape { server: (input) => Promise<Hooks> } (PluginModule).
 *
 * OpenCode loads this via: import("context-mode/plugin") → also supports
 * the named export ContextModePlugin for backward compat.
 * OpenCode 2 calls default.setup and registers tools/hooks on its context.
 *
 * Constraints:
 *   - No SessionStart hook (OpenCode doesn't support it — #14808, #5409)
 *   - context injection now via chat.system.transform surrogate (OC-1)
 *   - No routing file auto-write (avoid dirtying project trees)
 *   - Session cleanup happens at plugin init (no SessionStart)
 */

import z from "zod/v4";

import { dirname, resolve, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { existsSync, readFileSync } from "node:fs";

import { resolveSessionDbPath, SessionDB } from "../../session/db.js";
import { extractEvents, extractUserEvents, parseOpencodeUsage, buildAgentUsageEvent } from "../../session/extract.js";
import type { HookInput } from "../../session/extract.js";
import { buildResumeSnapshot } from "../../session/snapshot.js";
import type { SessionEvent } from "../../types.js";
import { AdapterPlatformType, OpenCodeAdapter } from "./index.js";
import { PLATFORM_ENV_VARS } from "../detect.js";
import { zod3ShapeToV4 } from "./zod3tov4.js";

// ── Types ─────────────────────────────────────────────────

/** KiloCode/OpenCode plugin input — both platforms pass at least `directory`. */
type PluginClientAppLogBodyExtra = {
  sessionId?: string;
  source?: string;
};

type PluginClientAppLogBody = {
  service: string;
  level: "info" | "warn" | "error" | "debug"; // Strict union for log levels
  message: string;
  extra?: PluginClientAppLogBodyExtra;
};

type PluginClientAppLogOptions = {
  body: PluginClientAppLogBody;
};

type PluginClientApp = {
  log: (options: PluginClientAppLogOptions) => Promise<void>;
};

type PluginClient = {
  app: PluginClientApp;
};

type PluginContext = {
  client: PluginClient;
  directory: string;
};

type NativeToolContext = {
  sessionID: string;
  messageID: string;
  agent: string;
  directory: string;
  worktree?: string;
  abort?: AbortSignal;
  metadata?: (input: { title?: string; metadata?: Record<string, unknown> }) => void;
};

type NativeToolDefinition = {
  description: string;
  args: Record<string, unknown>;
  execute: (
    args: Record<string, unknown>,
    ctx: NativeToolContext,
  ) => Promise<string | { title?: string; output: string; metadata?: Record<string, unknown> }>;
};

/** OpenCode tool.execute.before — first parameter */
interface BeforeHookInput {
  tool: string;
  sessionID: string;
  callID: string;
}

/** OpenCode tool.execute.before — second parameter */
interface BeforeHookOutput {
  args: any;
}

/** OpenCode tool.execute.after — first parameter */
interface AfterHookInput {
  tool: string;
  sessionID: string;
  callID: string;
  args: any;
}

/** OpenCode tool.execute.after — second parameter */
interface AfterHookOutput {
  title: string;
  output: string;
  metadata: any;
}

/**
 * OpenCode generic bus `event` hook — single parameter.
 * The plugin SDK delivers every bus Event here (refs/platforms/opencode/
 * packages/plugin/src/index.ts:224). We narrow to `message.updated`, whose
 * `properties.info` is the full assistant Message carrying tokens/cost/modelID.
 */
interface EventHookInput {
  event?: {
    type?: string;
    properties?: { info?: { sessionID?: string } & Record<string, unknown> };
  };
}

/** OpenCode experimental.session.compacting — first parameter */
interface CompactingHookInput {
  sessionID: string;
}

/** OpenCode experimental.session.compacting — second parameter */
interface CompactingHookOutput {
  context: string[];
  prompt?: string;
}

/**
 * OpenCode experimental.chat.system.transform — first parameter.
 * Verified against sst/opencode/dev/packages/plugin/src/index.ts:
 *   input: { sessionID?: string; model: Model }
 * `sessionID` is optional in the SDK type but is in practice always set
 * (the transform runs *for* a session). We treat it as required and
 * skip injection when absent rather than fall back to a fabricated ID.
 *
 * NOTE: We deliberately do NOT use `experimental.chat.messages.transform`.
 * Its SDK input shape is `{}` (no sessionID) and its output is
 * `{ messages: { info: Message; parts: Part[] }[] }` — the prior code
 * (`output.messages.unshift({ role, content })`) wrote a value of the
 * wrong shape and was silently dropped (Mickey / PR #376 root cause).
 */
interface SystemTransformHookInput {
  sessionID?: string;
  model: unknown;
}

/** OpenCode experimental.chat.system.transform — second parameter */
interface SystemTransformHookOutput {
  system: string[];
}

/**
 * OpenCode chat.message hook — verified against
 * refs/platforms/opencode/packages/plugin/src/index.ts:233.
 *   input:  { sessionID; agent?; model?; messageID?; variant? }
 *   output: { message: UserMessage; parts: Part[] }
 * We read text from `parts[*].text` (the orchestrator reference at
 * refs/plugin-examples/opencode/opencode-orchestrator/src/plugin-handlers/
 * chat-message-handler.ts:41-65 uses the same pattern).
 */
interface ChatMessageHookInput {
  sessionID: string;
  agent?: string;
  messageID?: string;
}

interface ChatMessagePart {
  type: string;
  text?: string;
}

interface ChatMessageHookOutput {
  message: unknown;
  parts: ChatMessagePart[];
}

// Synthetic message tags emitted by harnesses (CCv2 inline filter). When the
// user "message" is actually a system-generated nudge (e.g. tool-result, system
// reminder), capturing it as user_prompt would flood the DB with noise.
const SYNTHETIC_MESSAGE_PREFIXES = [
  "<task-notification>",
  "<system-reminder>",
  "<context_guidance>",
  "<tool-result>",
];

function isSyntheticMessage(text: string): boolean {
  const trimmed = text.trim();
  return SYNTHETIC_MESSAGE_PREFIXES.some((p) => trimmed.startsWith(p));
}

// ── Helpers ───────────────────────────────────────────────

// Quorum markers — must NOT be substrings of each other (#487).
// Each token uniquely identifies the routing block / context-mode rules
// without overlapping any other marker. The XML tag is the primary signal;
// the two distinctive bare tool names are the secondary signals. Together
// any 2 of 3 confirm the system prompt already carries routing instructions.
const ROUTING_MARKERS = [
  "<context_window_protection>",
  "ctx_search",
  "ctx_index",
];

function systemHasRoutingInstructions(system: string[]): boolean {
  const text = system.join("\n");
  // Word-boundary check guards against unrelated identifiers that happen to
  // share a prefix/suffix (e.g. a hypothetical `ctx_search_v2`).
  const wordBoundary = (m: string) => {
    if (m.startsWith("<")) return text.includes(m);
    const re = new RegExp(`(?:^|\\W)${m.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\$&")}(?:\\W|$)`);
    return re.test(text);
  };
  return ROUTING_MARKERS.filter(wordBoundary).length >= 2;
}

/**
 * Detect whether the plugin is running under KiloCode or OpenCode.
 *
 * Reuses the canonical PLATFORM_ENV_VARS list (src/adapters/detect.ts) instead
 * of hardcoding env var names — single source of truth, future-proof if Kilo
 * or OpenCode add/rename env vars upstream.
 *
 * Order matters: KiloCode is an OpenCode fork and sets `OPENCODE=1` in
 * addition to `KILO_PID`. PLATFORM_ENV_VARS lists `kilo` BEFORE `opencode`
 * so KILO_PID wins the iteration.
 *
 * Pre-fix version was `return process.env.KILO_PID ? "kilo" : "opencode";` —
 * surfaced by github.com/mksglu/context-mode/pull/376 (mikij). Full symmetric
 * fix: also actively check opencode env vars instead of blind fallback.
 */
function getPlatform(): AdapterPlatformType {
  for (const [platform, vars] of PLATFORM_ENV_VARS) {
    if (platform !== "kilo" && platform !== "opencode") continue;
    if (vars.some((v) => process.env[v.name])) {
      return platform as AdapterPlatformType;
    }
  }
  // Plugin host should always set one of the env vars. Fallback to opencode
  // (the wider ecosystem) when neither is set, for predictable behavior.
  return "opencode";
}

// Deliberately structural, narrow subsets of @opencode/plugin's public V2 API.
// Importing that package at runtime would break V1/Kilo consumers on import.
type SystemPart = { type: "text"; text: string; metadata?: Record<string, unknown> };
type ToolCall = { tool: string; sessionID: string; id: string; input: unknown };
type ToolResult = {
  content?: string | ReadonlyArray<{ type: string; text?: string }>;
  output?: unknown;
};
type ToolHooks = {
  "execute.before": ToolCall;
  "execute.after": ToolCall & (
    { status: "completed"; result: ToolResult } | { status: "error"; error: { message: string } }
  );
};
type SessionHooks = {
  prompt: { sessionID: string; prompt: { text: string } };
  context: { sessionID: string; system: SystemPart[] };
  compaction: { sessionID: string; system: SystemPart[] };
};
type V2ToolContext = {
  sessionID: string; messageID: string; agent: string; signal?: AbortSignal;
  progress(update: Record<string, unknown>): Promise<void>;
};
type V2Tool = {
  name: string; description: string; options: { namespace: string };
  input: z.ZodObject<Record<string, z.ZodType>>;
  execute(input: Record<string, unknown>, ctx: V2ToolContext): Promise<{
    content: string; metadata: { title: string };
  }>;
};
type V2Event = {
  id: string; type: string; location?: { directory: string }; data: unknown;
};
type V2Context = {
  location: { directory: string };
  tool: {
    transform(callback: (editor: { add(tool: V2Tool): void }) => void): Promise<unknown>;
    hook<K extends keyof ToolHooks>(name: K, callback: (event: ToolHooks[K]) => Promise<void>): Promise<unknown>;
  };
  session: {
    hook<K extends keyof SessionHooks>(name: K, callback: (event: SessionHooks[K]) => Promise<void>): Promise<unknown>;
  };
  event: { subscribe(options: { signal: AbortSignal }): AsyncIterable<V2Event> };
};
type Log = (message: string, extra?: PluginClientAppLogBodyExtra) => Promise<void>;

// Shared operations retain legacy behavior; only host payloads differ.
async function initializePlugin(projectDir: string, log: Log) {
  // Resolve build dir from compiled JS location
  const platform = getPlatform();
  const adapter = new OpenCodeAdapter(platform);
  const buildDir = dirname(fileURLToPath(import.meta.url));
  // initSecurity() looks for `<dir>/security.js`, which lives at the
  // top of build/ — two levels up from this adapter directory.
  const buildRoot = resolve(buildDir, "..", "..");

  // Load routing module (ESM .mjs, lives outside build/ in hooks/)
  const routingPath = resolve(buildDir, "..", "..", "..", "hooks", "core", "routing.mjs");
  const routing = await import(pathToFileURL(routingPath).href);
  await routing.initSecurity(buildRoot);

  // OC-1 / OC-3: Load hook helpers once at plugin init. Dynamic import keeps
  // the .mjs ESM islands isolated from the .ts compile graph.
  const routingBlockPath = resolve(buildDir, "..", "..", "..", "hooks", "routing-block.mjs");
  const routingBlockMod = await import(pathToFileURL(routingBlockPath).href);
  const toolNamingPath = resolve(buildDir, "..", "..", "..", "hooks", "core", "tool-naming.mjs");
  const toolNamingMod = await import(pathToFileURL(toolNamingPath).href);
  const autoInjectionPath = resolve(buildDir, "..", "..", "..", "hooks", "auto-injection.mjs");
  const autoInjectionMod = await import(pathToFileURL(autoInjectionPath).href);

  // Pre-build the routing block once per process — it is platform-specific
  // (tool naming differs between opencode and kilo) but does NOT depend on
  // sessionID, so we cache it. createToolNamer accepts both "opencode" and
  // "kilo" per hooks/core/tool-naming.mjs:25-26.
  const toolNamer = toolNamingMod.createToolNamer(platform);
  const routingBlock: string = routingBlockMod.createRoutingBlock(toolNamer);

  // Initialize per-process state. We do NOT fabricate a sessionId here —
  // OpenCode/Kilo provide the real `input.sessionID` on every hook, and a
  // process-global UUID would (a) never match prior-session resume rows and
  // (b) collide across multi-session reuse (Mickey / PR #376 root cause).
  // C2 narrowing: resolve DB path through the canonical helper directly.
  // BaseAdapter no longer exposes getSessionDBPath; the adapter only owns
  // the sessions DIR (per-platform), the helper owns the per-project FILE
  // (case-fold + worktree-suffix + one-shot legacy migration).
  const db = new SessionDB({
    dbPath: resolveSessionDbPath({ projectDir, sessionsDir: adapter.getSessionDir() }),
  });

  // Clean up old sessions on startup (no SessionStart hook to do this).
  db.cleanupOldSessions(7);

  // OC-4 (#487 follow-up): per-session capture gate. PR #487 trusted the host
  // to deliver AGENTS.md events, but OpenCode only fires `rule_content` events
  // when the user explicitly reads the file. snapshot.ts:172 + analytics.ts:152
  // CONSUME `rule_content` to render rules into the resume snapshot — without
  // this capture path, AGENTS.md is silently absent from continuity output.
  // Keyed by sessionId (NOT projectDir) so multi-session reuse within a long-
  // lived plugin process still gets per-session capture exactly once.
  const agentsMdCaptured = new Set<string>();

  /**
   * OC-4: Read AGENTS.md (with CLAUDE.md / CONTEXT.md fallbacks) from the
   * project directory and persist as `rule` + `rule_content` events. Mirrors
   * the CC SessionStart pattern at hooks/sessionstart.mjs:121-132 and the
   * OpenCode instruction.ts FILES order. Idempotent via `agentsMdCaptured`
   * Set keyed by sessionId. Fail-soft: missing/unreadable files do not throw.
   */
  function captureAgentsMd(sessionId: string): void {
    if (agentsMdCaptured.has(sessionId)) return;
    agentsMdCaptured.add(sessionId);
    const candidates = ["AGENTS.md", "CLAUDE.md", "CONTEXT.md"];
    for (const name of candidates) {
      try {
        const p = join(projectDir, name);
        if (!existsSync(p)) continue;
        const content = readFileSync(p, "utf-8");
        if (!content.trim()) continue;
        db.insertEvent(sessionId, {
          type: "rule",
          category: "rule",
          data: p,
          priority: 1,
        } as SessionEvent, "PluginInit");
        db.insertEvent(sessionId, {
          type: "rule_content",
          category: "rule",
          data: content,
          priority: 1,
        } as SessionEvent, "PluginInit");
      } catch {
        // file missing or unreadable — skip silently
      }
    }
  }

  function logger(
    message = "context-mode debug log",
    extra?: PluginClientAppLogBodyExtra,
  ): Promise<void> {
    return log(message, extra);
  }

  /**
   * Drop-in wrapper for `logger` that NEVER rejects (#448).
   *
   * The OPENCODE_DEBUG branch awaits `logger(...)` from inside the chat-turn
   * hot path (chat.system.transform). If `ctx.client.app.log` rejects —
   * transport error, closed stream, oversized payload — the promise rejection
   * propagates back to OpenCode core and can break the turn. Debug logging
   * is best-effort; swallow errors silently and let the turn proceed.
   */
  async function safeLog(
    message?: string,
    extra?: PluginClientAppLogBodyExtra,
  ): Promise<void> {
    try {
      await logger(message, extra);
    } catch {
      // Never break the turn on debug-log failure.
    }
  }

  async function buildNativeTools(): Promise<Record<string, NativeToolDefinition>> {
    // Import the existing MCP server registry without starting its stdio
    // transport. This is the plugin-only bridge for #574: OpenCode/Kilo
    // call ctx_* tools in-process through Hooks.tool instead of spawning
    // a separate MCP child per session.
    const prevEmbedded = process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS;
    process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS = "1";
    let mod: typeof import("../../server.js");
    try {
      mod = await import("../../server.js");
    } finally {
      if (prevEmbedded === undefined) delete process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS;
      else process.env.CONTEXT_MODE_EMBEDDED_PLUGIN_TOOLS = prevEmbedded;
    }
    const tools: Record<string, NativeToolDefinition> = {};

    for (const registered of mod.REGISTERED_CTX_TOOLS) {
      const config = registered.config as Record<string, unknown>;
      // Zod schema object that the MCP framework normally calls
      // safeParseAsync() on before invoking the handler. The native
      // OpenCode plugin path bypasses MCP's transport layer entirely
      // (refs/platforms/opencode/packages/opencode/src/tool/registry.ts:127),
      // so we must parse args here too — otherwise z.preprocess() coercions
      // (coerceCommandsArray / coerceJsonArray in server.ts) and defaults
      // never fire. Fixes #621.
      const inputSchema = config.inputSchema as
        | { shape?: unknown; _def?: { shape?: unknown }; parse?: (input: unknown) => unknown }
        | undefined;
      const shape =
        typeof inputSchema?.shape === "object" && inputSchema.shape !== null
          ? inputSchema.shape
          : typeof inputSchema?._def?.shape === "function"
            ? (inputSchema._def.shape as () => unknown)()
            : {};

      // Both KiloCode and recent OpenCode bundle Zod v4 in-host; v3 schemas
      // crash with `n._zod.def` undefined. Gate widened from kilo-only (#632)
      // because every consumer of this file is an OpenCode-family host.
      const argsForHost = zod3ShapeToV4(shape as Record<string, unknown>);

      tools[registered.name] = {
        description: String(config.description ?? ""),
        args: argsForHost,
        async execute(args: Record<string, unknown>, toolCtx: NativeToolContext) {
          toolCtx.metadata?.({ title: String(config.title ?? registered.name) });
          const project = toolCtx.directory || projectDir;

          // Run the registered Zod schema BEFORE the handler — same contract
          // as the MCP SDK (server/mcp.js safeParseAsync at line 174). This
          // applies z.preprocess() coercions, populates .default() values,
          // and produces the validation error the handler expects (#621).
          let parsedArgs: Record<string, unknown> = args ?? {};
          if (typeof inputSchema?.parse === "function") {
            try {
              parsedArgs = inputSchema.parse(args ?? {}) as Record<string, unknown>;
            } catch (err) {
              // Surface validation failures with a clear, actionable message
              // (mirrors MCP SDK error format) instead of a downstream
              // "x.map is not a function" crash.
              const message = err instanceof Error ? err.message : String(err);
              throw new Error(
                `Invalid arguments for ${registered.name}: ${message}`,
              );
            }
          }

          const result = await mod.withProjectDirOverride({ projectDir: project, sessionId: toolCtx.sessionID }, async () =>
            registered.handler(parsedArgs),
          );

          const r = result as {
            content?: Array<{ type?: string; text?: string }>;
            isError?: boolean;
          };
          const text = Array.isArray(r?.content)
            ? r.content
                .filter((c) => c?.type === "text" && typeof c.text === "string")
                .map((c) => c.text)
                .join("\n")
            : typeof result === "string"
              ? result
              : JSON.stringify(result ?? "");

          if (r?.isError) throw new Error(text || `${registered.name} returned an error`);
          return { title: String(config.title ?? registered.name), output: text };
        },
      };
    }

    return tools;
  }

  let nativeTools: Record<string, NativeToolDefinition>;
  try {
    nativeTools = await buildNativeTools();
  } catch (error) {
    db.close();
    throw error;
  }

  return {
    tool: nativeTools,
    db,

    // ── PreToolUse: Routing enforcement ─────────────────

    before: async (toolName: string, toolInput: Record<string, unknown>, sessionID?: string, nativeToolsAvailable = false) => {

      let decision;
      try {
        decision = routing.routePreToolUse(toolName, toolInput, projectDir, platform, sessionID, { nativeToolsAvailable });
      } catch {
        return; // Routing failure → allow passthrough
      }

      if (!decision) return; // No routing match → passthrough

      if (decision.action === "deny" || decision.action === "ask") {
        // Throw to block — OpenCode catches this and denies the tool call
        throw new Error(decision.reason ?? "Blocked by context-mode");
      }

      if (decision.action === "modify" && decision.updatedInput) {
        // Mutate output.args — OpenCode reads the mutated output object
        Object.assign(toolInput, decision.updatedInput);
      }

      if (decision.action === "context" && decision.additionalContext) {
        // Mutate output.args — OpenCode reads the mutated output object
        toolInput.additionalContext = decision.additionalContext;
      }
    },

    // ── PostToolUse: Session event capture ──────────────

    after: async (sessionId: string, toolName: string, args: Record<string, unknown>, response: string, isError?: boolean) => {
      if (!sessionId) return;
      try {
        db.ensureSession(sessionId, projectDir);
        // OC-4 (#487 follow-up): AGENTS.md → rule_content capture for snapshot
        // and auto-memory parity. Idempotent per-session via Set guard.
        captureAgentsMd(sessionId);

        const hookInput: HookInput = {
          tool_name: toolName,
          tool_input: args,
          tool_response: response,
          tool_output: isError === undefined ? undefined : { isError },
        };

        const events = extractEvents(hookInput);
        for (const event of events) {
          // Cast: extract.ts SessionEvent lacks data_hash (computed by insertEvent)
          db.insertEvent(sessionId, event as SessionEvent, "PostToolUse");
        }
      } catch {
        // Silent — session capture must never break the tool call
      }
    },

    // ── event: per-turn token + cost capture (paid-observability) ───
    // The generic bus `event` hook (refs/platforms/opencode/packages/plugin/
    // src/index.ts:224) delivers every Event; we filter `message.updated`
    // (published on each assistant-message update incl. step-finish —
    // session.ts:673) and read tokens/cost/modelID off properties.info
    // (assistant filter via role; refs stream.transport.ts:214-216).
    //
    // CAVEAT (refs processor.ts:717-718): message-level `.tokens` is the LAST
    // step's snapshot (overwritten per step-finish), while `.cost` is
    // cumulative for the turn. parseOpencodeUsage passes `.cost` through as
    // native_cost_usd so the billed $ stays exact despite the token snapshot
    // being last-step only. `message.updated` fires multiple times per turn;
    // because tokens are a terminal snapshot and cost is cumulative, the last
    // event for a message carries the final figures — re-emitting on each
    // update is idempotent at the cost column and merely refreshes the
    // last-step token telemetry. db.insertEvent both persists locally AND
    // forwards to the platform (the TS-plugin equivalent of the .mjs
    // attributeAndInsertEvents path).
    usage: async (sessionId: string, payload: unknown, identity?: string) => {
      try {
        if (!sessionId) return;
        const counts = parseOpencodeUsage(payload);
        if (!counts) return;
        const usageEvent = buildAgentUsageEvent(counts);
        if (!usageEvent) return;

        db.ensureSession(sessionId, projectDir);
        // Distinct V2 steps may have identical usage. Preserve their durable
        // identity so SessionDB's content dedup does not discard valid charges.
        if (identity) usageEvent.data += ` model:${counts.model_id} step:${identity}`;
        db.insertEvent(sessionId, usageEvent, identity ? "StepEnded" : "MessageUpdated");
      } catch {
        // Silent — usage capture must never break the session.
      }
    },

    // ── chat.message: User-prompt capture (OC-2 / Z2) ───
    // SDK signature verified at refs/platforms/opencode/packages/plugin/src/
    // index.ts:233. Orchestrator reference at refs/plugin-examples/opencode/
    // opencode-orchestrator/src/plugin-handlers/chat-message-handler.ts:41-65.
    // CCv2 inline filter: skip synthetic harness messages (system reminders,
    // tool results, etc.) so we don't pollute the user-prompt event stream.
    prompt: async (sessionId: string, message: string) => {
      if (!sessionId) return;
      try {
        if (!message || isSyntheticMessage(message)) return;

        db.ensureSession(sessionId, projectDir);
        // OC-4 (#487 follow-up): also capture on chat.message so sessions that
        // never invoke a tool still seed rule_content events for continuity.
        captureAgentsMd(sessionId);

        // 1. Always save the raw prompt
        db.insertEvent(sessionId, {
          type: "user_prompt",
          category: "user-prompt",
          data: message,
          priority: 1,
        } as SessionEvent, "UserPromptSubmit");

        // 2. Extract role/decision/intent/skill events from the prompt body
        const userEvents = extractUserEvents(message);
        for (const ev of userEvents) {
          db.insertEvent(sessionId, ev as SessionEvent, "UserPromptSubmit");
        }
      } catch {
        // Silent — chat.message must never break the turn
      }
    },

    // ── PreCompact: Snapshot generation ─────────────────

    compact: async (sessionId: string, context: string[]) => {
      if (!sessionId) return "";
      try {
        db.ensureSession(sessionId, projectDir);
        const events = db.getEvents(sessionId);
        if (events.length === 0) return "";

        const stats = db.getSessionStats(sessionId);
        const snapshot = buildResumeSnapshot(events, {
          compactCount: (stats?.compact_count ?? 0) + 1,
        });

        db.upsertResume(sessionId, snapshot, events.length);
        db.incrementCompactCount(sessionId);

        // Mutate output.context to inject the snapshot
        context.push(snapshot);

        if (process.env.OPENCODE_DEBUG) {
          await safeLog(snapshot, {
            sessionId,
            source: "on compaction - snapshot",
          });
        }

        // OC-3 / Z3: Add budget-capped auto-injection (P1 role / P2 rules /
        // P3 skills / P4 intent — ≤500 tokens / ~2000 chars per
        // hooks/auto-injection.mjs). Pushed as a separate context entry so
        // OpenCode can fold it independently from the verbose snapshot.
        try {
          const autoBlock: string = autoInjectionMod.buildAutoInjection(events);
          if (autoBlock && autoBlock.length > 0) {
            context.push(autoBlock);
          }

          if (process.env.OPENCODE_DEBUG) {
            await safeLog(autoBlock, {
              sessionId,
              source: "on compaction - autoBlock",
            });
          }
        } catch {
          // Auto-injection failure must NOT break the snapshot path.
        }

        return snapshot;
      } catch {
        return "";
      }
    },

    // ── SessionStart equivalent (PR #376) ───────────────
    // OpenCode lacks a real SessionStart hook (#14808, #5409). The closest
    // surrogate is `experimental.chat.system.transform` — verified shape:
    //   input:  { sessionID?: string; model: Model }
    //   output: { system: string[] }
    // We claim the most-recent unconsumed resume snapshot atomically (race-
    // safe across concurrent processes) and prepend it to the system prompt.
    system: async <Part>(sessionId: string | undefined, system: Part[], text: (part: Part) => string, part: (text: string) => Part) => {
      if (!sessionId) return;

      // ── OC-1 / CCv1: ROUTING_BLOCK injection ──────────────
      // Inject the <context_window_protection> XML block on the first
      // chat.system.transform per session. This is INDEPENDENT of the
      // resume snapshot path below — routing block must fire even when
      // no prior session row exists. Splice at index 1 (NOT unshift) for
      // the same OpenCode llm.ts:117-128 cache-fold reason as resume.
      //
      // Skip injection when system prompt already contains context-mode
      // routing rules (e.g. via AGENTS.md / CLAUDE.md loaded by the host).
      // Detect by checking for a quorum of distinctive tool names — any two
      // of ctx_execute, ctx_batch_execute, ctx_fetch_and_index confirms the
      // instructions are present and avoids ~2K chars of duplication.
      if (Array.isArray(system)) {
        if (!systemHasRoutingInstructions(system.map(text))) {
          try {
            system.splice(1, 0, part(routingBlock));
          } catch {
            // Never break the chat turn on routing-block injection failure.
          }

          if (process.env.OPENCODE_DEBUG) {
            await safeLog(text(system[1]), {sessionId, source: 'on routing block injection'});
          }
        } else if (process.env.OPENCODE_DEBUG) {
          await safeLog(`routing block skipped — system prompt already contains context-mode instructions`, {sessionId, source: 'on routing block injection'});
        }
      }

      try {
        // Pass current sessionId so SQL excludes self-injection (v1.0.106 — Mickey #376
        // follow-up): if Session B compacts mid-flight and produces its own row,
        // B's next system.transform must NOT claim that row back into B's prompt.
        const row = db.claimLatestUnconsumedResume(sessionId);
        if (!row || !row.snapshot) return;        // no row → retry on next turn

        if (process.env.OPENCODE_DEBUG) {
          await safeLog(row.snapshot, {
            sessionId,
            source: "on resume - snapshot",
          });
        }

        if (Array.isArray(system)) {
          // Insert at index 1 (after the header) — NOT unshift.
          // OpenCode's llm.ts:117-128 saves `header = system[0]` BEFORE this
          // hook runs and then folds the rest into a 2-part structure
          // `[header, body]` only if `system[0] === header` after the hook.
          // Prepending via unshift replaces system[0] with the snapshot,
          // making the equality check fail → cache-fold is skipped → every
          // system block is sent as a separate `role: "system"` message →
          // provider prompt cache is invalidated on every resume injection.
          // Inserting at index 1 keeps the header invariant and lets the
          // snapshot ride along inside the cached body block.
          system.splice(1, 0, part(row.snapshot));
          // Mark consumed only AFTER successful splice so failed paths can retry
          if (process.env.OPENCODE_DEBUG) {
            await safeLog(text(system[1]), { sessionId, source: "on resume" });
          }
        }
      } catch {
        // Silent — never break the chat turn
      }
    },
  };
}

/** The legacy public factory and hook signatures remain callable unchanged. */
async function createContextModePlugin(ctx: PluginContext) {
  const runtime = await initializePlugin(ctx?.directory ?? process.cwd(), (message, extra) =>
    ctx.client.app.log({ body: { service: "context-mode-logger", level: "info", message, extra } }));
  return {
    tool: runtime.tool,
    "tool.execute.before": async (input: BeforeHookInput, output: BeforeHookOutput) => {
      // Preserve V1's process-based routing-throttle identity.
      await runtime.before(input.tool ?? "", output.args ?? {});
    },
    "tool.execute.after": async (input: AfterHookInput, output: AfterHookOutput) => {
      await runtime.after(input.sessionID, input.tool ?? "", input.args ?? {}, output.output);
    },
    event: async (input: EventHookInput) => {
      // V1 message.updated: tokens are the last-step snapshot; cost is the
      // cumulative message cost. Preserve the existing parser's semantics.
      const event = input?.event;
      if (event?.type === "message.updated" && event.properties?.info?.sessionID) {
        await runtime.usage(event.properties.info.sessionID, event);
      }
    },
    "chat.message": async (input: ChatMessageHookInput, output: ChatMessageHookOutput) => {
      const text = output?.parts?.find((part) => part?.type === "text" && part.text)?.text;
      if (text) await runtime.prompt(input?.sessionID, text);
    },
    "experimental.session.compacting": async (input: CompactingHookInput, output: CompactingHookOutput) => {
      return runtime.compact(input.sessionID, output.context);
    },
    "experimental.chat.system.transform": async (input: SystemTransformHookInput, output: SystemTransformHookOutput) => {
      await runtime.system(input?.sessionID, output.system, (text) => text, (text) => text);
    },
  };
}

const V2_TOOL_ALIASES: Record<string, string> = {
  shell: "Bash", read: "Read", write: "Write", edit: "Edit", grep: "Grep",
  glob: "Glob", webfetch: "WebFetch", subagent: "Agent", question: "AskUserQuestion",
};
const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" ? value as Record<string, unknown> : {};
const textPart = (text: string): SystemPart => ({ type: "text", text });
function resultText(result: ToolResult): string {
  if (typeof result.content === "string") return result.content;
  if (Array.isArray(result.content)) return result.content.filter((part) => part.type === "text").map((part) => part.text ?? "").join("\n");
  return typeof result.output === "string" ? result.output : JSON.stringify(result.output ?? "");
}

async function setupOpenCodeV2(ctx: V2Context): Promise<() => Promise<void>> {
  const projectDir = ctx.location.directory;
  const runtime = await initializePlugin(projectDir, async (message, extra) => console.info("context-mode:", message, extra ?? ""));
  const controller = new AbortController();
  let stream: Promise<void> | undefined;
  let cleanupPromise: Promise<void> | undefined;
  const cleanup = () => cleanupPromise ??= (async () => {
    controller.abort();
    try { await stream; } finally { runtime.db.close(); }
  })();
  try {
    await ctx.tool.transform((editor) => {
      for (const [name, tool] of Object.entries(runtime.tool)) editor.add({
        name, description: tool.description, options: { namespace: "context-mode" },
        input: z.object(tool.args as Record<string, z.ZodType>),
        async execute(input, context) {
          const result = await tool.execute(input, { ...context, directory: projectDir, abort: context.signal });
          return { content: typeof result === "string" ? result : result.output,
            metadata: { title: typeof result === "string" ? name : result.title ?? name } };
        },
      });
    });
    await ctx.tool.hook("execute.before", async (event) => {
      // Other plugins may use primitive/array schemas. Do not replace their
      // inputs on passthrough; our routed builtins use object arguments.
      if (event.input === null || typeof event.input !== "object" || Array.isArray(event.input)) return;
      const input = record(event.input);
      await runtime.before(V2_TOOL_ALIASES[event.tool] ?? event.tool, input, event.sessionID, true);
    });
    await ctx.tool.hook("execute.after", async (event) => {
      const input = { ...record(event.input) };
      if (input.path !== undefined && input.file_path === undefined) input.file_path = input.path;
      await runtime.after(event.sessionID, V2_TOOL_ALIASES[event.tool] ?? event.tool, input,
        event.status === "completed" ? resultText(event.result) : event.error.message,
        event.status === "error" ? true : undefined);
    });
    await ctx.session.hook("prompt", async (event) => runtime.prompt(event.sessionID, event.prompt.text));
    await ctx.session.hook("context", async (event) => {
      // Quorum markers use bare ctx_* names; V2 exposes namespaced tool IDs.
      // Its generated Code Mode catalog lists tool names but is NOT routing
      // guidance. Exclude that part from the quorum's read-only view.
      await runtime.system(event.sessionID, event.system,
        (part) => part.text.startsWith("# Code Mode\n") ? "" : part.text.replaceAll("context-mode_ctx_", "ctx_"), textPart);
    });
    await ctx.session.hook("compaction", async (event) => {
      // A new native cycle starts with a fresh system array. Reprocessing this
      // operation must not increment counts/inject twice; ordinary mentions of
      // session_resume in user rules are NOT evidence of a previous injection.
      if (event.system.some((part) => part.metadata?.source === "context-mode-compaction")) return;
      const context: string[] = [];
      await runtime.compact(event.sessionID, context);
      for (const text of context) event.system.push({ ...textPart(text), metadata: { source: "context-mode-compaction" } });
      // Do not assign event.result: that would SKIP OpenCode's native summary.
    });

    // session.usage.updated is cumulative across the SESSION, not a delta.
    // Correlate per-step model metadata with ended tokens/cost instead.
    const models = new Map<string, { modelID?: string; providerID?: string }>();
    stream = (async () => {
      try {
        for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
          if (controller.signal.aborted) break;
          if (event.type !== "session.step.started" && event.type !== "session.step.ended") continue;
          const data = record(event.data);
          if (typeof data.sessionID !== "string" || typeof data.assistantMessageID !== "string") continue;
          // Public events can be from other locations. An explicit foreign
          // location overrides historical membership (e.g. a session moved).
          if (event.location) {
            if (resolve(event.location.directory) !== resolve(projectDir)) continue;
          } else if (!runtime.db.getSessionStats(data.sessionID)) continue;
          const key = data.assistantMessageID;
          if (event.type === "session.step.started") {
            const model = record(data.model);
            models.set(key, {
              modelID: typeof model.id === "string" ? model.id : undefined,
              providerID: typeof model.providerID === "string" ? model.providerID : undefined,
            });
          } else {
            const model = models.get(key);
            // Only an admitted start can settle once. The public stream is
            // live (not a replay feed); deleting also bounds completed state.
            if (!model) continue;
            models.delete(key);
            await runtime.usage(data.sessionID, { ...model, role: "assistant", tokens: data.tokens, cost: data.cost }, event.id);
          }
          // Failed/interrupted steps may never end; bound their metadata.
          if (models.size > 500) models.delete(models.keys().next().value!);
        }
      } catch (error) {
        if (!controller.signal.aborted) console.error("context-mode: event subscription failed", error);
      }
    })();
    return cleanup;
  } catch (error) {
    await cleanup();
    throw error;
  }
}

export default {
  id: "context-mode",
  server: createContextModePlugin,
  setup: setupOpenCodeV2,
};
export { createContextModePlugin as ContextModePlugin };
// Test surface for the quorum substring fix (#487).
export { systemHasRoutingInstructions, ROUTING_MARKERS };
