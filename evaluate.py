from datasets import Dataset
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from ragas.run_config import RunConfig

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0,
    google_api_key=gemini_api_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=gemini_api_key
)

data = {
    "user_input": [
        "How performance evaluation is done?",
        "What is the Employement term at SoftwareTree?"
    
    ],
    "retrieved_contexts": [
        ["Clause 14: Performance Evaluaton The Employees performance shall be reviewed periodically by SoftwareTree. Performance reviews shall be based on role-specific goals and conduct. Unsatsfactory performance may impact increments or contnuation of employment."],
        ["EMPLOYMENT AGREEMENT is entered into between SoftwareTree and the Employee for the purpose of defining the terms and conditons of employment. This document is created solely for educatonal and demonstraton purposes.  Clause 1: Employment Term Clause 1: Employment Term The Employee is appointed by SoftwareTree for a fixed employment term of two (2) years, commencing from the official date of joining."]
    ],
    "response": [
        "Performance reviews shall be based on role-specific goals and conduct.",
        "The fixed employment term at SoftwareTree is two (2) years, commencing from the official date of joining."
    ],
    "reference": [
        "Performance Evaluation of the employee's performance shall be reviewed periodically. It shall be based on role-specific goals and conduct.",
        "Employement term is 2 years which is fixed and starts from the day of joining."
    ]
}
dataset = Dataset.from_dict(data)


result = evaluate(
    dataset,
    metrics=[answer_relevancy, faithfulness, context_precision, context_recall],
    llm=llm,
    embeddings=embeddings
)

print("RAGAS Evaluation Results")
print(result)