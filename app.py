import boto3
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify
import json
import re
import os

app = Flask(__name__)

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2",
)


# Function to download PDF from URL
def download_pdf_from_url(pdf_url):
    try:
        loader = PyPDFLoader(pdf_url)
        pdf_contents = loader.load()
        return pdf_contents
    except Exception as e:
        print(f"Error downloading PDF from URL: {e}")
        return None

def extract_insurance_info(pdfcontents):
    prompt = """Extract below entities from below text, and produce output in specified format.
        Example: For text like this "Insured Name: John Doe
        Registration Number: ABC123
        Make/Model: Toyota Camry
        Period of Insurance: 01/01/2024 - 01/01/2025
        Engine Number: 1234567890
        Chassis No: ABC123456DEF789
        Policy Number: XYZ789"
        Extract the following entities:
        Entities: ```{Entities}```
        Format:```{format}```
        text:```{pdfcontents}```
        """
    format = "dict"
    Entities = """
        {
        "Insured Name": value,
        "Registration Number": value,
        "Make/Model": value,
        "Period of Insurance": value,
        "Engine Number": value,
        "Chassis No": value,
        "Policy Number": value
        }
        """
    entity_template = PromptTemplate(input_variables=["Entities", "format", "pdfcontents"], template=prompt)
    entity_ext = entity_template.format(pdfcontents=pdfcontents, Entities=Entities, format=format)
    
    return entity_ext

def bedrock_runtime_stream(prompt):
    final_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"

    parameters = {
        "prompt": final_prompt,
        "max_tokens_to_sample": 600,
        "temperature": 0,
        "top_k": 10
    }

    invoke_model_kwargs = {
        "body": json.dumps(parameters),
        "accept": '*/*',
        "contentType": 'application/json',
        "modelId": 'anthropic.claude-v2'
    }

    res = bedrock_runtime.invoke_model(**invoke_model_kwargs)
    res_body = json.loads(res.get('body').read())
    query = res_body['completion'].replace("python", "").strip()
    extracted_values = re.findall(r'\{([^}]*)\}', query)
    return extracted_values

@app.route('/docqna', methods=["POST"])
def processclaim():
    try:
        header_key = request.headers.get('X-API-Key')
        if header_key == os.environ.get('X-API-Key'):
            input_json = request.get_json(force=True)
            pdf_url = input_json["pdf_url"]
            pdfcontents = download_pdf_from_url(pdf_url)
            prompt = extract_insurance_info(pdfcontents)
            extracted_values = bedrock_runtime_stream(prompt)
            response = json.loads("{"+ extracted_values[0] + "}")
            return response
        else:
            return jsonify({"Status": "Failure --- invalid X-API-key"})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"Status": "Failure --- some error occurred"})