"use client";

import type { ComponentProps, ReactNode } from "react";

// `linkifyCitations` (both chat and research-report) emits citation links
// shaped as `[\[N\]](url)`. After streamdown parses that, the anchor's
// visible text is the literal string `[N]`. We use that shape to
// distinguish citation anchors from regular markdown links so we only
// apply the superscript styling where it belongs.
const CITATION_PATTERN = /^\[\d+\]$/;

function getTextContent(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") {
    return "";
  }
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(getTextContent).join("");
  }
  if (typeof node === "object" && "props" in node) {
    return getTextContent(
      (node as { props?: { children?: ReactNode } }).props?.children
    );
  }
  return "";
}

type CitationAnchorProps = ComponentProps<"a"> & { node?: unknown };

export function CitationAnchor({
  href,
  children,
  className,
  node: _node,
  ...rest
}: CitationAnchorProps) {
  const isCitation =
    !!href && CITATION_PATTERN.test(getTextContent(children).trim());

  if (isCitation) {
    return (
      <sup className="citation-ref">
        <a
          href={href}
          rel="noopener noreferrer"
          target="_blank"
        >
          {children}
        </a>
      </sup>
    );
  }

  return (
    <a className={className} href={href} {...rest}>
      {children}
    </a>
  );
}
