import { textblockTypeInputRule } from "prosemirror-inputrules";
import { type MarkSpec, Schema } from "prosemirror-model";
import { schema } from "prosemirror-schema-basic";
import { addListNodes } from "prosemirror-schema-list";
import type { Transaction } from "prosemirror-state";
import type { EditorView } from "prosemirror-view";
import type { MutableRefObject } from "react";

import { buildContentFromDocument } from "./functions";

// Citation markers (`[N]`) are wrapped in <sup class="citation-ref"> by
// the rehype pipeline. Without a matching mark in the schema, ProseMirror's
// DOMParser would strip the wrapper and we'd lose the superscript styling
// inside the editable artifact. Registering `sup` as a mark preserves it
// across parse/render; the markdown serializer (see editor/functions.tsx)
// emits no syntax for it, so the on-disk markdown stays as bare `[N]`.
const supMark: MarkSpec = {
  parseDOM: [{ tag: "sup" }],
  toDOM: () => ["sup", { class: "citation-ref" }, 0],
};

export const documentSchema = new Schema({
  nodes: addListNodes(schema.spec.nodes, "paragraph block*", "block"),
  marks: schema.spec.marks.addToEnd("sup", supMark),
});

export function headingRule(level: number) {
  return textblockTypeInputRule(
    new RegExp(`^(#{1,${level}})\\s$`),
    documentSchema.nodes.heading,
    () => ({ level })
  );
}

export const handleTransaction = ({
  transaction,
  editorRef,
  onSaveContent,
}: {
  transaction: Transaction;
  editorRef: MutableRefObject<EditorView | null>;
  onSaveContent: (updatedContent: string, debounce: boolean) => void;
}) => {
  if (!editorRef?.current) {
    return;
  }

  const newState = editorRef.current.state.apply(transaction);
  editorRef.current.updateState(newState);

  if (transaction.docChanged && !transaction.getMeta("no-save")) {
    const updatedContent = buildContentFromDocument(newState.doc);

    if (transaction.getMeta("no-debounce")) {
      onSaveContent(updatedContent, false);
    } else {
      onSaveContent(updatedContent, true);
    }
  }
};
