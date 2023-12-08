import os, sys
import json
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from .utils import check_path
# import xmltodict

# open the input xml file and read
# data in form of python dictionary 
# using xmltodict module

def convert_to_mcscript_statement(qa_file: str, output_file: str):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    check_path(output_file)

    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for line in tqdm(qa_handle, total=nrow):
            json_line = json.loads(line)
            output_dict = convert_qajson_to_entailment(json_line)
            output_handle.write(json.dumps(output_dict))
            output_handle.write("\n")
    print(f'converted statements saved to {output_file}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict):
    question_text = qa_json["question"]["stem"]
    choices = qa_json["question"]["choices"]
    for choice in choices:
        choice_text = choice["text"]
        statement = question_text + ' ' + choice_text
        create_output_dict(qa_json, statement, choice["label"] == qa_json.get("answerKey", "A"))

    return qa_json


# Create the output json dictionary from the input json, premise and hypothesis statement
def create_output_dict(input_json: dict, statement: str, label: bool) -> dict:
    if "statements" not in input_json:
        input_json["statements"] = []
    input_json["statements"].append({"label": label, "statement": statement})
    return input_json