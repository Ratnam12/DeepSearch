"use client";

import { motion } from "framer-motion";
import Image from "next/image";

// Empty-state landing for the chat. Big logo + wordmark + a single tag
// line, all centered. Suggested actions render in their own row above
// the input box (see multimodal-input.tsx) so this stays focused on the
// product identity — Perplexity-style minimalism rather than a packed
// dashboard.

export const Greeting = () => {
  return (
    <div
      className="pointer-events-auto flex flex-col items-center gap-5 px-4"
      key="overview"
    >
      <motion.div
        animate={{ opacity: 1, y: 0, scale: 1 }}
        className="relative"
        initial={{ opacity: 0, y: 12, scale: 0.96 }}
        transition={{ delay: 0.15, duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      >
        <Image
          alt="DeepSearch"
          className="size-14 dark:hidden"
          height={56}
          priority
          src="/images/deepsearch-light.png"
          width={56}
        />
        <Image
          alt="DeepSearch"
          className="hidden size-14 dark:block"
          height={56}
          priority
          src="/images/deepsearch-dark.png"
          width={56}
        />
      </motion.div>
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="text-center font-medium text-3xl tracking-[-0.02em] text-foreground md:text-[42px] md:leading-[1.1]"
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.3, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        What do you want to research?
      </motion.div>
      <motion.div
        animate={{ opacity: 1, y: 0 }}
        className="text-center text-[13px] text-muted-foreground/70 md:text-sm"
        initial={{ opacity: 0, y: 10 }}
        transition={{ delay: 0.45, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        Ask anything — grounded answers with citations and follow-up sources.
      </motion.div>
    </div>
  );
};
