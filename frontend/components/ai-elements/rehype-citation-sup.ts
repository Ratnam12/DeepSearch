// Minimal hast types so we don't need a direct `@types/hast` dep
// (the streamdown chain pulls it in transitively but pnpm doesn't
// hoist it to the top level).
type HText = { type: "text"; value: string };
type HElement = {
  type: "element";
  tagName: string;
  properties?: Record<string, unknown>;
  children: HChild[];
};
type HChild =
  | HText
  | HElement
  | { type: string; [key: string]: unknown };
type HRoot = { type: "root"; children: HChild[] };

// Bare `[N]` citation markers (the kind a freeform text artifact ends up
// with — no URL behind them) get wrapped in <sup class="citation-ref">
// so they read as academic-style footnote markers instead of inline
// body text. Linked citations (`<a href="...">[N]</a>`) are left for
// CitationAnchor at React render time, so we skip text inside <a> here.
const CITATION_RE = /[ \t]?\[(\d+)\]/g;
const SKIP_TAGS = new Set(["a", "sup", "code", "pre"]);

function makeSupNode(num: string): HElement {
  return {
    type: "element",
    tagName: "sup",
    properties: { className: ["citation-ref"] },
    children: [{ type: "text", value: `[${num}]` }],
  };
}

function splitTextNode(node: HText): HChild[] | null {
  const text = node.value;
  CITATION_RE.lastIndex = 0;
  if (!CITATION_RE.test(text)) {
    return null;
  }
  CITATION_RE.lastIndex = 0;

  const out: HChild[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null = CITATION_RE.exec(text);
  while (match !== null) {
    const before = text.slice(cursor, match.index);
    if (before) {
      out.push({ type: "text", value: before });
    }
    out.push(makeSupNode(match[1]));
    cursor = match.index + match[0].length;
    match = CITATION_RE.exec(text);
  }
  if (cursor < text.length) {
    out.push({ type: "text", value: text.slice(cursor) });
  }
  return out;
}

function walk(parent: HElement | HRoot): void {
  const next: HChild[] = [];
  for (const child of parent.children) {
    if (child.type === "element") {
      const elem = child as HElement;
      if (!SKIP_TAGS.has(elem.tagName)) {
        walk(elem);
      }
      next.push(elem);
      continue;
    }
    if (child.type === "text") {
      const replacement = splitTextNode(child as HText);
      if (replacement) {
        next.push(...replacement);
        continue;
      }
    }
    next.push(child);
  }
  parent.children = next;
}

export function rehypeCitationSup() {
  return (tree: HRoot) => {
    walk(tree);
  };
}
