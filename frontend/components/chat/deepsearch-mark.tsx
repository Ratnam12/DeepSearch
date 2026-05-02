"use client";

import Image from "next/image";
import { cn } from "@/lib/utils";

export function DeepSearchMark({
  size = 16,
  className,
  priority = false,
}: {
  size?: number;
  className?: string;
  priority?: boolean;
}) {
  return (
    <>
      <Image
        alt="DeepSearch"
        className={cn("dark:hidden", className)}
        height={size}
        priority={priority}
        src="/images/deepsearch-light.png"
        width={size}
      />
      <Image
        alt=""
        aria-hidden
        className={cn("hidden dark:block", className)}
        height={size}
        priority={priority}
        src="/images/deepsearch-dark.png"
        width={size}
      />
    </>
  );
}
