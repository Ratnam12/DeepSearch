CREATE TABLE IF NOT EXISTS "UserCredits" (
	"userId" varchar(255) PRIMARY KEY NOT NULL,
	"email" varchar(320),
	"bonusChat" integer DEFAULT 0 NOT NULL,
	"bonusDeepSearch" integer DEFAULT 0 NOT NULL,
	"notes" text,
	"createdAt" timestamp DEFAULT now() NOT NULL,
	"updatedAt" timestamp DEFAULT now() NOT NULL
);
