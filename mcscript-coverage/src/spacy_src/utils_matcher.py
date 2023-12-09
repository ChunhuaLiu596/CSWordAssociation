# from matcher_span_single_test import lemmatize
import spacy
from spacy.tokens import Doc
from spacy.matcher import Matcher,PhraseMatcher
from spacy.vocab import Vocab
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd

from utils import flat_list

# nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
nlp = spacy.blank("en")


def get_lemmas(x):
    x = x.replace("_", " ")
    doc = nlp(x)
    # tokens = [tok.text for tok in doc]
    lemmas = [{"LEMMA": tok.lemma_} for tok in doc]
    return lemmas

def get_orths(x):
    x = x.replace("_", " ")
    doc = nlp.make_doc(x)
    # tokens = [tok.text for tok in doc]
    orths = [{"ORTH": tok.orth_} for tok in doc]
    return orths


def generate_lemma_pattern_pair(h, t, lemmatized=True):
    matcher = Matcher(nlp.vocab)
    middle = [{'OP': '*', 'LENGTH': {"<=": 10}}]

    if lemmatized: 
        pattern =  [{"LEMMA": lemma} for lemma in h] +  middle + [{"LEMMA": lemma} for lemma in t] 
        label = "_".join([" ".join(h) , " ".join(t)])
    else:
        pattern = get_lemmas(h) + middle + get_lemmas(t)
        label = "_".join([h,t])

    matcher.add(label, [pattern])
    return matcher


def generate_orth_pattern_pair(h, t):
    matcher = Matcher(nlp.vocab)
    middle = [{'OP': '*', 'LENGTH': {"<=": 10}}]

    pattern = get_orths(h) + middle + get_orths(t)
    label = "_".join([h,t])

    matcher.add(label, [pattern])
    return matcher

def generate_orth_pattern_concept(h):
    matcher = Matcher(nlp.vocab)
    pattern = get_orths(h) 
    # label = "_".join([h])
    label=h
    matcher.add(label, [pattern])
    return matcher


def generate_pattern_phrase(h):
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add(h, [nlp.make_doc(h)])

    return matcher

# class _PretokenizedTokenizer:
#     """Custom tokenizer to be used in spaCy when the text is already pretokenized."""
#     def __init__(self, vocab: Vocab):
#         """Initialize tokenizer with a given vocab
#         :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
#         """
#         self.vocab = vocab

#     def __call__(self, inp: Union[List[str], str]) -> Doc:
#         """Call the tokenizer on input `inp`.
#         :param inp: either a string to be split on whitespace, or a list of tokens
#         :return: the created Doc object
#         """
#         if isinstance(inp, str):
#             words = inp.split()
#             spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
#             return Doc(self.vocab, words=words, spaces=spaces)
#         elif isinstance(inp, list):
#             return Doc(self.vocab, words=inp)
#         else:
#             raise ValueError("Unexpected input format. Expected string to be split on whitespace, or list of tokens.")

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        # words = text.split(" ")
        return Doc(self.vocab, words=words)

# nlp = spacy.blank("en")

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

# nlp.tokenizer = _PretokenizedTokenizer(nlp.vocab)
def nlp_chunk(texts, batch_size=200):
    '''
    load the tokenized texts
    '''
    # nlp.tokenizer = nlp.tokenizer.tokens_from_list
    # nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
    preproc_pipe = []
    # doc = Doc(nlp.vocab, words=words,)

    # for doc in tqdm(Doc(nlp.vocab, words=texts)):
    # for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total = len(texts)):
        # preproc_pipe.append(doc)
    # for doc in tqdm(nlp.pipe(*texts, batch_size=batch_size), total = len(texts)):
        # preproc_pipe.append(doc)
    for text in tqdm(texts,  total = len(texts)):
        preproc_pipe.append(nlp.pipe(text, batch_size=batch_size))
    return preproc_pipe

def nlp_parallel(texts, total_length, n_jobs=20, chunksize=100):
    ''' 
    1. chunk texts by chunksize 
    2. assign each chunk to multiple jobs
    3. each chunk is lemmatizes with a batch 

    return a list of nlp object
    '''
    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(nlp_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, total_length, chunksize=chunksize))
    result = executor(tasks)
    # return flatten(result)
    return flat_list(result)






def make_docs(path, debug):
    # docs = nlp(df[])
    if debug:
        df = pd.read_json(path,orient='records',lines=True, nrows=5)
    else:
        df = pd.read_json(path,orient='records',lines=True)

    # df['nlp'] = df['clean'].apply(lambda x: nlp.pipe(x))
    # df['nlp']= list(nlp.pipe(df['clean'], batch_size=256, n_process=20, disable=["ner", "parser"]))
    df['nlp']=nlp_parallel(df['clean'], total_length=len(df.index), chunksize=50000)
    if debug:
        print (df.head(3))
    return df
