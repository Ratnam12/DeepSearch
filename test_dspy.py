import time
import logging
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

from backend.dspy_modules import decompose_query, synthesize_answer

QUESTION = "What causes inflation?"
CONTEXTS = (
    "[1] src=https://example.com\n"
    "Inflation is caused by increased money supply.\n\n"
    "[2] src=https://example2.com\n"
    "Demand-pull inflation occurs when demand exceeds supply."
)

# ── Step 1: Query decomposition ──────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Query Decomposition")
print("="*60)
print(f"  Input question : {QUESTION}")
print("  Calling DSPy Predict(DecomposeQuery)...")
t0 = time.time()

decomp = decompose_query(question=QUESTION)

print(f"  Done in {time.time()-t0:.2f}s")
print(f"\n  Raw LM prompt sent by DSPy:")
print("-"*60)
# dspy.settings.lm.history[-1] holds the last call
try:
    import dspy
    last = dspy.settings.lm.history[-1]
    for msg in last.get("messages", []):
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        print(f"  [{role}]\n{content}\n")
except Exception:
    print("  (history not available)")
print("-"*60)
print(f"\n  Raw LM response:")
print(f"  {decomp.queries}")

# ── Step 2: Answer Synthesis ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Answer Synthesis")
print("="*60)
print(f"  Contexts passed in:\n{CONTEXTS}\n")
print("  Calling DSPy Predict(SynthesizeAnswer)...")
t0 = time.time()

synthesis = synthesize_answer(question=QUESTION, contexts=CONTEXTS)

print(f"  Done in {time.time()-t0:.2f}s")
print(f"\n  Raw LM prompt sent by DSPy:")
print("-"*60)
try:
    last = dspy.settings.lm.history[-1]
    for msg in last.get("messages", []):
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        print(f"  [{role}]\n{content}\n")
except Exception:
    print("  (history not available)")
print("-"*60)

# ── Final output ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULT — Answer")
print("="*60)
print(synthesis.answer)
print("\n" + "-"*60)
print("RESULT — Citations")
print("-"*60)
print(synthesis.citations)
print("="*60 + "\n")
