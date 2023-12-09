
import pandas as pd 
import spacy 


path = 'output/mcscript2.en.csv'


df = pd.read_csv(path, names=['rel', 'head', 'tail', 'weight'])
df['head_len'] = df['head'].apply(lambda x: len(eval(x)))

df.query('words_len>3')


nlp = spacy.load('en_core_web_sm', disable=[ 'parser', 'ner'])

def extract_noun_verb():
    '''
    single verb, nouns
    compound noun
    phrasal verb
    '''

    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total = len(texts)):
        preproc_pipe.append([tok.pos_ for pos in doc])
