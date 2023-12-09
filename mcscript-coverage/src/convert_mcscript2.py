# %%
import os, sys
import json
import xmltodict
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

"""
2. output_file:
   {
   "id": "d3b479933e716fb388dfb297e881054c",
   "question": {
      "stem": "If a lantern is not for sale, where is it likely to be?"
      "choices": [{"label": "A", "text": "antique shop"}, {"label": "B", "text": "house"}, {"label": "C", "text": "dark place"}]
    },
    "answerKey":"B",

    "statements":[
        {label:true, stem: "If a lantern is not for sale, it likely to be at house"}, 
        {label:false, stem: "If a lantern is not for sale, it likely to be at antique shop"}, 
        {label:false, stem: "If a lantern is not for sale, it likely to be at dark place"}
        ]
  }
"""


# %%

def convert_xml_text(input_path, output_path):
    with open(input_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    texts = []
    ids = []
    scenarios = []
    questions = []
    choices = []
    json_data = {}
    for k in data_dict['data']['instance']:
        texts.append(k['text'].lower())
        ids.append(k['@id'])
        scenarios.append(k['@scenario'])
        # print(k)
        json_data = {}
        # json_data[] = 

        choices = []
        for item in k['questions']['question']:
            item_id = item['@id']
            question_text = item['@text']

            for ans in item['answer']:
                choice_text = ans['@text']
                choice_label = ans['@correct']

                choices.append({"label":choice_label, "text": choice_text})
        print(choices)
        # json_data = {
        #     "id": "d3b479933e716fb388dfb297e881054c",
        #     "question": {
        #         "stem": question_text, 
        #         "choices": [{"label": "A", "text": "antique shop"}, {"label": "B", "text": "house"}, {"label": "C", "text": "dark place"}]
        #         },
        #         "answerKey":"B",

        #         "statements":[
        #             {label:true, stem: "If a lantern is not for sale, it likely to be at house"}, 
        #             {label:false, stem: "If a lantern is not for sale, it likely to be at antique shop"}, 
        #             {label:false, stem: "If a lantern is not for sale, it likely to be at dark place"}
        #             ]
        #     }
           
            # print(" ")

    
    df = pd.DataFrame({ "id": ids,
                        "scenario": scenarios,
                        "text": texts 
                        }, columns=['id', 'scenario', 'text']) 

    df.to_csv(output_path)
    print(f"save {output_path} {len(df.index)} lines")
    return df


# %%
input_path = '../data/MCScript2.0/dev-data.xml'
output_path = '../output/MCScript2.0/dev-data.xml'
convert_xml_text(input_path, output_path)

# %%
OrderedDict([
    ('@id', '0'), 
    ('@scenario', 'renovating a room'), 
    ('text', "Last week , my girlfriend and I decided to renovate our basement . We wanted to make it look new and fresh . First , we took pictures of it and spent some time looking at the pictures , deciding what we wanted to keep and what things we wanted to change about the room . We then made a list of things we needed to buy at the store to accomplish this renovation . We came back home and began working in the basement . We worked together and removed certain things first . Then , we painted the basement walls together . Then , we hung pictures on the walls that we 'd bought . We then placed a few items like a couch and large TV that we 'd bought in the basement . We added our new carpets too . Finally , we put the popcorn machine in the corner of the basement and our basement looked new and fresh ! Our basement had been nicely renovated indeed !"), 
    ('questions', 
        OrderedDict([('question', 
                [OrderedDict([('@id', '0'), ('@text', 'Where did they go with the list?'), ('@type', 'text'), ('answer', [OrderedDict([('@correct', 'False'), ('@id', '0'), ('@text', 'The supermarket.')]), OrderedDict([('@correct', 'True'), ('@id', '1'), ('@text', 'to the store')])])]), 
                
                 OrderedDict([('@id', '1'), ('@text', 'What was bought at the store?'), ('@type', 'text'), ('answer', [OrderedDict([('@correct', 'False'), ('@id', '0'), ('@text', 'a new laptop')]), OrderedDict([('@correct', 'True'), ('@id', '1'), ('@text', 'paint, couch, TV, carpets, pictures')])])]), 
                
                OrderedDict([('@id', '2'), ('@text', 'Why did they wait until the paint was dried?'), ('@type', 'commonsense'), ('answer', [OrderedDict([('@correct', 'True'), ('@id', '0'), ('@text', 'So they could hang the pictures up')]), OrderedDict([('@correct', 'False'), ('@id', '1'), ('@text', 'So they could put down the carpets')])])])])]))])

# %%


# %%


import xml.etree.ElementTree as ET

def convert_xml_to_json(input_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    json_data = []
    count = 0
    for instance in root.iter('instance'):
        data ={}
        # print(instance.tag, instance.attrib)
        p_id = instance.attrib['id']
        for p in instance.iter('text'):
            p_text = p.text 

        for qs in instance.iter('questions'):
            for q in qs:
                q_text = q.attrib['text']
                stem = p_text + ' [SEP] ' + q_text 
                q_id = q.attrib['id']
                id = "-".join([p_id, q_id])
                data['id']= id 

                question = {}
                question['stem'] = stem 
                question['q_text'] = q_text 
                question['p_text'] = p_text 
                choices=[]
            
                for ans in q.iter('answer'):
                    ans_dict = {}
                    label = chr(int(ans.attrib['id'])+ 65)      
                    ans_dict["text"] = ans.attrib['text']
                    ans_dict["label"] = label 
                    choices.append(ans_dict)
                    
                    if ans.attrib["correct"] == "True":
                        answerKey=label

                print(id, label, answerKey)
                question['choices'] = choices 
                question['answerKey'] = answerKey 
                data['question'] = question
                json_data.append(data)
    return json_data 

# %%
import pandas as pd 
data_dir = '../data/MCScript2.0/'
names = ['train-data.xml', 'dev-data.xml', 'test-data.xml']

for name in names:
    input_path = os.path.join(data_dir, name)
    output_path = os.path.join(data_dir, name.split(".")[0]+'.json')

    json_data = convert_xml_to_json(input_path)

    df = pd.DataFrame(json_data)
    df.to_json(output_path, orient='records', lines=True)
    print(f"save {output_path} {len(df.index)} lines")

# %%

# all_data = []
# for i, child in enumerate(root):
#     data = dict()
#     # print(child.tag, child.attrib, child.text)
#     text = child[0].text
#     # print(text)
#     questions = child[1]
#     data['question'] = {}
    
#     data['question']['choices'] = []
    
#     for elem in child[1]:
#         data['question']['stem'] = elem[0]
#         print(elem[0].tag)
#         for x in elem[1:]:
#             tag = x.tag
#             attrib = x.attrib 
#             print(tag, attrib)
        # print (elem.tag, elem.text, elem.attrib)
        # if elem.text !={}:
        #     data[elem.tag] = elem.text
        # for gchild in elem:
        #       print (elem.tag, elem.text, elem.attrib)

        # if elem.text != {} and elem.attrib == {} :
        #     data[elem.tag] = elem.text 
            
        # if  elem.attrib != {} :
        #     data[elem.tag] = elem.attrib 
            
    # all_data.append(data)

# for data in all_data[0]:
# for k, v in all_data[0].items():
#     print(k,v)

#     all_tag = [(elem.tag, elem.attrib, elem.text) for elem in child.iter()]
#     data.append(all_tag)
# for x in all_tag:
#     print(x)
# print(all_tag)
# print(root.tag, root.attrib)

# for x in myroot[0]:
    #  print(x.tag,  x.text)