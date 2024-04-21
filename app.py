import os
import tempfile
import requests
import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.indexes import VectorstoreIndexCreator
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from flask import Flask, request, jsonify

app = Flask(__name__)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_runtime,
    region_name="us-west-2",
)

llm = Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock_runtime, region_name="us-west-2")

# Function to download PDF from URL
def download_pdf_from_url(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise error for invalid response
        return response.content
    except Exception as e:
        print(f"Error downloading PDF from URL: {e}")
        return None

def load_pdf_and_texts(pdf_content):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        chunk_size_value = 1000
        chunk_overlap = 100
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap, length_function=len)
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

def initialize_docembeddings(pdf_content):
    texts = load_pdf_and_texts(pdf_content)
    if texts:
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=FAISS,
            embedding=embeddings,
        )
        index_from_documents = index_creator.from_documents(texts)
        return index_from_documents.vectorstore
    return None

def get_docembeddings(pdf_url):
    pdf_content = download_pdf_from_url(pdf_url)
    if pdf_content:
        tmp_index_file = f'/tmp/{pdf_url.replace("/", "_").replace(":", "_")}.faiss'
        if not os.path.exists(tmp_index_file):
            docembeddings = initialize_docembeddings(pdf_content)
            if docembeddings:
                docembeddings.save_local(tmp_index_file)
        else:
            docembeddings = FAISS.load_local(tmp_index_file, embeddings, allow_dangerous_deserialization=True)
        return docembeddings
    else:
        return None

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

Begin!

Context:
---------
{context}
---------
Question: {questions}
Helpful Answer:"""

output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "questions"],
    output_parser=output_parser
)
# Load QA chain
chain = load_qa_chain(llm, chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)

    
@app.route('/docqna', methods=["POST"])
def processclaim():
    try:
        input_json = request.get_json(force=True)
        pdf_url = input_json["pdf_url"]
        queries = input_json["queries"]
        combined_query = " ".join(queries)  # Combine queries into a single string
        docembeddings = get_docembeddings(pdf_url)
        if docembeddings:
            relevant_chunks = docembeddings.similarity_search_with_score(combined_query, k=2)
            chunk_docs = [chunk[0] for chunk in relevant_chunks]
            results = chain({"input_documents": chunk_docs, "questions": combined_query})
            return jsonify(results["output_text"])
        else:
            return jsonify({"Status": "Failure --- Unable to load document embeddings"})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"Status": "Failure --- some error occurred"})

if __name__ == "__main__":
    app.run(debug=True)