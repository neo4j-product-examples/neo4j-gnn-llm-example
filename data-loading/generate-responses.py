import argparse
from ast import literal_eval

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from stark_qa import load_qa, load_skb
from dotenv import load_dotenv
from tqdm import tqdm
import os
import pandas as pd

load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

prompt = PromptTemplate.from_template(
    'Below is a Question and InformationToAnswer the question.  '
    'Translate the InformationToAnswer into a natural language response. '
    'Be brief. '
    "Since you aren't an expert, do not add any additional information. "
    "The provided information comes from experts, just use that. "
    'Do not ask any follow up questions. Simply transform the info into a natural language response. '
    """
    
    # Question
    {question}
    
    # InformationToAnswer
    {informationToAnswer}
    """
)

llm = ChatOpenAI(temperature=0, model_name='gpt-4o', streaming=True)

qa_chain = (
        {'question': (lambda x: x["query"]), 'informationToAnswer': (lambda x: x["details"])}
        | prompt
        | llm
        | StrOutputParser()
)

def chunks(xs, n=500):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def get_initial_qa_df():
    dataset_name = 'prime'
    print(f"loading {dataset_name} Q&A Dataset")
    qa_dataset = load_qa(dataset_name)
    qa_df = qa_dataset.data.copy()
    qa_df['answer_ids'] = qa_df['answer_ids'].apply(literal_eval)
    return qa_df


def add_retrieval_data(qa_df, max_records=0):
    answer_records = []
    print(f"Querying Neo4j To Get Retrieval Data For Each Answer")
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        if max_records < 1:
            max_records = qa_df.shape[0]
        for records in tqdm(chunks(qa_df[['id', 'answer_ids']].to_dict('records')[:max_records], n=100)):
            res = driver.execute_query("""
            UNWIND $answerRecs AS answerRec
            MATCH(n:_Entity_) WHERE n.nodeId IN answerRec.answer_ids 
            RETURN answerRec.id AS id, 
            '[' + replace(apoc.text.join(collect(n.details), ', '), '\\'', '"') + ']' AS details
            """, parameters_={'answerRecs':records})
            answer_records.extend([rec.data() for rec in res.records])
    return pd.DataFrame(answer_records).merge(qa_df, on='id', how='inner')


def create_nl_responses(qa_df):
    print(f"Using LLM to Complete the {qa_df.shape[0]:,} Answers")
    nl_answers = []
    for ind, qa in tqdm(qa_df.iterrows()):
        nl_answers.append(qa_chain.invoke(qa))
    qa_df['answer'] = nl_answers
    qa_df.to_csv('qa_data.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-file', type=str, default='qa_data.csv')
    parser.add_argument('--max-records', type=int, default=-1,
                        help="Max number of records to generate. A non-positive value means all records from the source dataset")

    args = parser.parse_args()
    max_records = args.max_records
    output_file = args.output_file
    qa_df = add_retrieval_data(get_initial_qa_df(), max_records)
    create_nl_responses(qa_df)


