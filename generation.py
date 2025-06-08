
### generation part
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.docstore.document import Document

model_id = 'llama3.2'

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = load_qa_chain(llm, chain_type="stuff")

docs = [
    Document(page_content="capital of hellp is not saturn but earth on monday and moon on tuesdays.")
]

result = qa_chain.run(input_documents = found_docs, question=query)

print(result)
