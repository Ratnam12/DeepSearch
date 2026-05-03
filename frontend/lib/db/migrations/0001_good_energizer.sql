CREATE TABLE IF NOT EXISTS "ResearchEvent" (
	"runId" uuid NOT NULL,
	"seq" integer NOT NULL,
	"type" varchar(64) NOT NULL,
	"payload" json NOT NULL,
	"ts" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "ResearchEvent_runId_seq_pk" PRIMARY KEY("runId","seq")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ResearchPlan" (
	"runId" uuid NOT NULL,
	"version" integer NOT NULL,
	"briefMd" text,
	"subQuestions" json NOT NULL,
	"outline" json NOT NULL,
	"approvedAt" timestamp,
	"createdAt" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "ResearchPlan_runId_version_pk" PRIMARY KEY("runId","version")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ResearchReport" (
	"runId" uuid NOT NULL,
	"version" integer NOT NULL,
	"markdown" text NOT NULL,
	"sections" json,
	"finalizedAt" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "ResearchReport_runId_version_pk" PRIMARY KEY("runId","version")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ResearchRun" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"userId" varchar(255) NOT NULL,
	"query" text NOT NULL,
	"status" varchar(32) DEFAULT 'queued' NOT NULL,
	"modelSettings" json,
	"error" text,
	"createdAt" timestamp DEFAULT now() NOT NULL,
	"startedAt" timestamp,
	"finishedAt" timestamp
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ResearchSource" (
	"runId" uuid NOT NULL,
	"citationNum" integer NOT NULL,
	"url" text NOT NULL,
	"title" text,
	"snippet" text,
	"fetchedAt" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "ResearchSource_runId_citationNum_pk" PRIMARY KEY("runId","citationNum")
);
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS "ResearchSubagent" (
	"runId" uuid NOT NULL,
	"id" varchar(64) NOT NULL,
	"subQuestion" text NOT NULL,
	"status" varchar(32) DEFAULT 'queued' NOT NULL,
	"model" varchar(128),
	"findingMd" text,
	"sources" json,
	"startedAt" timestamp,
	"finishedAt" timestamp,
	"createdAt" timestamp DEFAULT now() NOT NULL,
	CONSTRAINT "ResearchSubagent_runId_id_pk" PRIMARY KEY("runId","id")
);
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "ResearchEvent" ADD CONSTRAINT "ResearchEvent_runId_ResearchRun_id_fk" FOREIGN KEY ("runId") REFERENCES "public"."ResearchRun"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "ResearchPlan" ADD CONSTRAINT "ResearchPlan_runId_ResearchRun_id_fk" FOREIGN KEY ("runId") REFERENCES "public"."ResearchRun"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "ResearchReport" ADD CONSTRAINT "ResearchReport_runId_ResearchRun_id_fk" FOREIGN KEY ("runId") REFERENCES "public"."ResearchRun"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "ResearchSource" ADD CONSTRAINT "ResearchSource_runId_ResearchRun_id_fk" FOREIGN KEY ("runId") REFERENCES "public"."ResearchRun"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
DO $$ BEGIN
 ALTER TABLE "ResearchSubagent" ADD CONSTRAINT "ResearchSubagent_runId_ResearchRun_id_fk" FOREIGN KEY ("runId") REFERENCES "public"."ResearchRun"("id") ON DELETE cascade ON UPDATE no action;
EXCEPTION
 WHEN duplicate_object THEN null;
END $$;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "ResearchRun_user_idx" ON "ResearchRun" USING btree ("userId","createdAt");--> statement-breakpoint
CREATE INDEX IF NOT EXISTS "ResearchRun_status_idx" ON "ResearchRun" USING btree ("status");