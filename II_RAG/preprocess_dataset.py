import os
import json
import pandas as pd
from collections import defaultdict
from RAG.I_constants import source_path

squad_path = './squad/train-00000-of-00001.parquet'

os.makedirs(source_path, exist_ok=True) # make sure that we hace the path exist
def extract_questions_and_answers(squad_parquet_path):
    df_org = pd.read_parquet(squad_parquet_path) # read the dataset
    question_context_map = defaultdict(set)

    for context, question in zip(df_org['context'], df_org['question']):
        question_context_map[question].add(context)

    questions = []
    for n, q in enumerate(question_context_map):
        if len(question_context_map[q])>1:
            new_question = {'text': q, 'sources':[]}
            for m, c in enumerate(question_context_map[q]):
                new_id = "Q%d_C%d.txt"%(n, m)
                new_question['sources'].append(new_id)
                with open(os.path.join(source_path, new_id), 'w', encoding='utf-8') as wr:
                    wr.writelines(c + "\n")
            questions.append(new_question)
    with open('./squad/squad_multiple_contexts.json', 'w') as wr:
        json.dump(questions, wr, indent=1)

if __name__ == "__main__":
    extract_questions_and_answers(squad_path)