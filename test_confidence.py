from backend.agent import check_confidence

# Test 1: empty list → should return None
print(check_confidence([]))

# Test 2: low scores → should return refusal string
low_chunks = [{"rerank_score": 0.2}, {"rerank_score": 0.3}]
print(check_confidence(low_chunks))

# Test 3: good scores → should return None
good_chunks = [{"rerank_score": 0.9}, {"rerank_score": 0.7}]
print(check_confidence(good_chunks))