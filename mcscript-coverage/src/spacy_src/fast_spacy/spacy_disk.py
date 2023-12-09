import spacy
from spacy.tokens import DocBin
import pandas as pd
from tqdm import tqdm
import functools
import operator
from joblib import Parallel, delayed

def load_disk(path):
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs

def load_multiple_disk(paths):
    docs = []
    for path in tqdm(paths, total=len(paths), desc='loading corpus docbin'):
        docs.extend(load_disk(path))
    return docs

def flat_list(a):
    return functools.reduce(operator.iconcat, a, [])

def load_parallel_disk(paths):
    chunks = paths 
    n_jobs = len(chunks)
    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(load_disk)
    tasks = (do(chunk) for chunk in chunks)
    result = executor(tasks)
    # result = flatten(result)
    result = flat_list(result)
    return result

def test():
    # path="data/test_disk"
    
    # doc_bin = DocBin(attrs=["ORTH","POS","LEMMA"], store_user_data=True)
    # texts = ["Some text", "Lots of texts...", "stripes is a noun", "He stripes him"]
    
    
    # nlp = spacy.load("en_core_web_sm")
    # for doc in nlp.pipe(texts):
    #     doc_bin.add(doc)
    # bytes_data = doc_bin.to_disk(path)
    
    # df=pd.DataFrame({"clean":list(texts)})
    
    # # # Deserialize later, e.g. in a new process
    # # nlp = spacy.blank("en")
    # # doc_bin = DocBin().from_disk(path)
    # # docs = list(doc_bin.get_docs(nlp.vocab))
    
    
    # # print(df)
    # nlp = spacy.load('en_core_web_sm')
    # df['docs']=df['clean'].apply(nlp)
    # df.to_csv(path+".csv")
    
    
    
    # path="data/test"
    # path = 'data/bookcorpus/clean_lemma_files/nlp_serialize'
    path = 'data/bookcorpus/clean_lemma_files/nlp_serialize'
    nlp = spacy.blank("en")
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    print(docs[:4])
    df=pd.DataFrame({"docs": docs})
    
    for doc in df['docs'].head(10):
        for t in doc:
            print(t.text, t.pos_, t.lemma_,)
    

if __name__=='__init__':
    test()