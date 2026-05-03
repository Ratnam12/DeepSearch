import type { InferSelectModel } from "drizzle-orm";
import {
  boolean,
  foreignKey,
  index,
  integer,
  json,
  pgTable,
  primaryKey,
  text,
  timestamp,
  uuid,
  varchar,
} from "drizzle-orm/pg-core";

// User identity is owned by Clerk — there is no local `User` table. Every
// row that previously had a uuid `userId` foreign key now stores Clerk's
// string user id (e.g. "user_2abc...") in a varchar column.

export const chat = pgTable("Chat", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  createdAt: timestamp("createdAt").notNull().defaultNow(),
  title: text("title").notNull(),
  userId: varchar("userId", { length: 255 }).notNull(),
  visibility: varchar("visibility", { enum: ["public", "private"] })
    .notNull()
    .default("private"),
});

export type Chat = InferSelectModel<typeof chat>;

export const message = pgTable("Message_v2", {
  id: uuid("id").primaryKey().notNull().defaultRandom(),
  chatId: uuid("chatId")
    .notNull()
    .references(() => chat.id),
  role: varchar("role").notNull(),
  parts: json("parts").notNull(),
  attachments: json("attachments").notNull(),
  createdAt: timestamp("createdAt").notNull().defaultNow(),
});

export type DBMessage = InferSelectModel<typeof message>;

export const vote = pgTable(
  "Vote_v2",
  {
    chatId: uuid("chatId")
      .notNull()
      .references(() => chat.id),
    messageId: uuid("messageId")
      .notNull()
      .references(() => message.id),
    isUpvoted: boolean("isUpvoted").notNull(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.chatId, table.messageId] }),
  })
);

export type Vote = InferSelectModel<typeof vote>;

export const document = pgTable(
  "Document",
  {
    id: uuid("id").notNull().defaultRandom(),
    createdAt: timestamp("createdAt").notNull().defaultNow(),
    title: text("title").notNull(),
    content: text("content"),
    kind: varchar("text", { enum: ["text", "code", "image", "sheet"] })
      .notNull()
      .default("text"),
    userId: varchar("userId", { length: 255 }).notNull(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.id, table.createdAt] }),
  })
);

export type Document = InferSelectModel<typeof document>;

export const suggestion = pgTable(
  "Suggestion",
  {
    id: uuid("id").notNull().defaultRandom(),
    documentId: uuid("documentId").notNull(),
    documentCreatedAt: timestamp("documentCreatedAt").notNull(),
    originalText: text("originalText").notNull(),
    suggestedText: text("suggestedText").notNull(),
    description: text("description"),
    isResolved: boolean("isResolved").notNull().default(false),
    userId: varchar("userId", { length: 255 }).notNull(),
    createdAt: timestamp("createdAt").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.id] }),
    documentRef: foreignKey({
      columns: [table.documentId, table.documentCreatedAt],
      foreignColumns: [document.id, document.createdAt],
    }),
  })
);

export type Suggestion = InferSelectModel<typeof suggestion>;

export const stream = pgTable(
  "Stream",
  {
    id: uuid("id").notNull().defaultRandom(),
    chatId: uuid("chatId").notNull(),
    createdAt: timestamp("createdAt").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.id] }),
    chatRef: foreignKey({
      columns: [table.chatId],
      foreignColumns: [chat.id],
    }),
  })
);

export type Stream = InferSelectModel<typeof stream>;

// ─── DeepSearch research-agent tables ────────────────────────────────────
//
// Long-running, durable research runs (5–30 min) with sub-agent
// orchestration, plan approval, and live event streaming. These tables
// back POST /api/research and the /research/[id] reader page. The
// `ResearchEvent` log is both the live progress feed (one row per agent
// step) and the durable history that survives refresh + server restart.

// Allowed values for ResearchRun.status. Kept as a const array so the
// worker (Python) and the frontend/queries (TypeScript) can stay in sync
// — both sides import from this list.
export const RESEARCH_RUN_STATUSES = [
  "queued",
  "scoping",
  "planning",
  "awaiting_approval",
  "researching",
  "writing",
  "done",
  "failed",
  "cancelled",
] as const;

export type ResearchRunStatus = (typeof RESEARCH_RUN_STATUSES)[number];

export const researchRun = pgTable(
  "ResearchRun",
  {
    id: uuid("id").primaryKey().notNull().defaultRandom(),
    userId: varchar("userId", { length: 255 }).notNull(),
    query: text("query").notNull(),
    status: varchar("status", { length: 32 })
      .notNull()
      .default("queued")
      .$type<ResearchRunStatus>(),
    modelSettings: json("modelSettings"),
    error: text("error"),
    createdAt: timestamp("createdAt").notNull().defaultNow(),
    startedAt: timestamp("startedAt"),
    finishedAt: timestamp("finishedAt"),
  },
  (table) => ({
    userIdx: index("ResearchRun_user_idx").on(table.userId, table.createdAt),
    statusIdx: index("ResearchRun_status_idx").on(table.status),
  })
);

export type ResearchRun = InferSelectModel<typeof researchRun>;

export type ResearchSubQuestion = {
  id: string;
  question: string;
  rationale?: string;
};

export type ResearchOutlineSection = {
  id: string;
  title: string;
  description?: string;
};

export const researchPlan = pgTable(
  "ResearchPlan",
  {
    runId: uuid("runId")
      .notNull()
      .references(() => researchRun.id, { onDelete: "cascade" }),
    version: integer("version").notNull(),
    briefMd: text("briefMd"),
    subQuestions: json("subQuestions")
      .notNull()
      .$type<ResearchSubQuestion[]>(),
    outline: json("outline").notNull().$type<ResearchOutlineSection[]>(),
    approvedAt: timestamp("approvedAt"),
    createdAt: timestamp("createdAt").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.runId, table.version] }),
  })
);

export type ResearchPlan = InferSelectModel<typeof researchPlan>;

// Append-only event log. `seq` is allocated atomically per run by the
// worker (SELECT max(seq) + 1 inside the same transaction as the INSERT)
// so the SSE endpoint can replay deterministically using `Last-Event-ID`.
export const researchEvent = pgTable(
  "ResearchEvent",
  {
    runId: uuid("runId")
      .notNull()
      .references(() => researchRun.id, { onDelete: "cascade" }),
    seq: integer("seq").notNull(),
    type: varchar("type", { length: 64 }).notNull(),
    payload: json("payload").notNull(),
    ts: timestamp("ts").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.runId, table.seq] }),
  })
);

export type ResearchEvent = InferSelectModel<typeof researchEvent>;

export const researchSubagent = pgTable(
  "ResearchSubagent",
  {
    runId: uuid("runId")
      .notNull()
      .references(() => researchRun.id, { onDelete: "cascade" }),
    id: varchar("id", { length: 64 }).notNull(),
    subQuestion: text("subQuestion").notNull(),
    status: varchar("status", { length: 32 }).notNull().default("queued"),
    model: varchar("model", { length: 128 }),
    findingMd: text("findingMd"),
    sources: json("sources"),
    startedAt: timestamp("startedAt"),
    finishedAt: timestamp("finishedAt"),
    createdAt: timestamp("createdAt").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.runId, table.id] }),
  })
);

export type ResearchSubagent = InferSelectModel<typeof researchSubagent>;

// Citations are deduped per run. The same URL surfaced by two sub-agents
// gets one citation number; sub-agents reference sources by URL and the
// writer renders the deduped numbering.
export const researchSource = pgTable(
  "ResearchSource",
  {
    runId: uuid("runId")
      .notNull()
      .references(() => researchRun.id, { onDelete: "cascade" }),
    citationNum: integer("citationNum").notNull(),
    url: text("url").notNull(),
    title: text("title"),
    snippet: text("snippet"),
    fetchedAt: timestamp("fetchedAt").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.runId, table.citationNum] }),
  })
);

export type ResearchSource = InferSelectModel<typeof researchSource>;

export const researchReport = pgTable(
  "ResearchReport",
  {
    runId: uuid("runId")
      .notNull()
      .references(() => researchRun.id, { onDelete: "cascade" }),
    version: integer("version").notNull(),
    markdown: text("markdown").notNull(),
    sections: json("sections"),
    finalizedAt: timestamp("finalizedAt").notNull().defaultNow(),
  },
  (table) => ({
    pk: primaryKey({ columns: [table.runId, table.version] }),
  })
);

export type ResearchReport = InferSelectModel<typeof researchReport>;
