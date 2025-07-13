from hf_qa import query_huggingface_qa

question = "What is the capital of France?"
context = "France is a country in Western Europe. Its capital is Paris."

try:
    response = query_huggingface_qa(question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {response.get('answer')}")
    print(f"Score: {response.get('score')}")
except Exception as e:
    print(f"An error occurred: {e}")


