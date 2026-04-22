from backend.dspy_modules import synthesize_answer
import logging
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

result = synthesize_answer(
    question="What causes inflation?",
    contexts="[1] src=https://example.com\nInflation is caused by increased money supply.\n\n[2] src=https://example2.com\nDemand-pull inflation occurs when demand exceeds supply."
)

print("\n" + "="*60)
print("✅  ANSWER")
print("="*60)
print(result.answer)
print("\n" + "-"*60)
print("📎  CITATIONS")
print("-"*60)
print(result.citations)
print("="*60 + "\n")