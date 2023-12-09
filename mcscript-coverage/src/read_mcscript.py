import os, sys
import json
import xmltodict
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import spacy 
from spacy.lang.en import English


# from tok import sent_tokenize #removed -> rthemove ? 

# open the input xml file and read
# data in form of python dictionary 
# using xmltodict module

def convert_xml_text(input_path, output_path):
    with open(input_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    texts = []
    ids = []
    scenarios = []
    for k in data_dict['data']['instance']:
        texts.append(k['text'].lower())
        ids.append(k['@id'])
        scenarios.append(k['@scenario'])
    
    df = pd.DataFrame({ "id": ids,
                        "scenario": scenarios,
                        "text": texts 
                        }, columns=['id', 'scenario', 'text']) 

    df.to_csv(output_path)
    print(f"save {output_path} {len(df.index)} lines")
    return df

def spacy_sents(doc, nlp):
    sents = []
    for x in nlp(doc).sents:
        sents.append(x.text)
        # print(x)
    # out = [' '.join(sents)]
    # print(out)
    
    return sents

def sent_tokenization(df):
    nlp = English()
    nlp.add_pipe("sentencizer")
   
    df['sent'] = df['text'].progress_apply(lambda x: spacy_sents(x, nlp))
    # df['sent'] = df['text'].progress_apply(lambda x: [ ' '.join(y) for y in nlp(x).sents])

    # df['sent'] = df['text'].progress_apply(lambda x: [ ' '.join(y) for y in sent_tokenize(x)])
    return df

def aggregate_all_sentences(df):
    '''
    aggregate sentences from all scenarios to a whole new DataFrame
    '''
    sents_all = set()
    for x in df['sent']:
        for y in x:
            sents_all.add(y.replace("^\ n", ""))

    df_new = pd.DataFrame(list(sents_all), columns=['sent'])
    # print(df_new.head(10))
    return df, df_new

def main(output_path=None, N=None, aggregate_sent=False):
    root = './data/mcscript2/'
    root_out = './output/mcscript2/'

    dfs = []
    for prefix in ['dev-data.xml', 'test-data.xml', 'train-data.xml']:
        df = convert_xml_text(os.path.join(root, prefix),os.path.join(root_out, prefix+".csv") )
        dfs.append(df)

    df = pd.concat(dfs).drop_duplicates()
    if N is not None:
        df = df.iloc[[int(N)]]
        print(df)

    df= sent_tokenization(df)

    if output_path is None:
        output_path = os.path.join(root_out, 'mcscript2.csv') 
    else:
        output_path = os.path.join(root_out, output_path) 

    if aggregate_sent:
        df_new = aggregate_all_sentences(df)
        df_new.to_csv(output_path, index=False)
        print(f"save {output_path} {len(df_new.index)} lines")
    else: 
        df.to_csv(output_path, index=False)
        print(f"save {output_path} {len(df.index)} lines")


if __name__=='__main__':
    output_path = sys.argv[1]
    N = None
    if len(sys.argv) ==3:
        N = sys.argv[2]
    main(output_path, N)
