# %%
"""
## Background
Consider you have a large text dataset on which you want to apply some non-trivial NLP transformations, such as stopword removal followed by lemmatizing the words  (i.e. reducing them to root form) in the text. [spaCy](https://spacy.io/usage) is an industrial strength NLP library designed for just such a task.

In the example shown below, the [New York Times dataset](https://www.kaggle.com/nzalake52/new-york-times-articles) is used to showcase how to significantly speed up a spaCy NLP pipeline. The goal is to take in an article's text, and speedily return a list of lemmas with unnecessary words, i.e. *stopwords*, removed.

Pandas DataFrames provide a convenient interface to work with tabular data of this nature. First, import the necessary modules shown.
"""

# %%
#collapse-hide
import re
import pandas as pd
import spacy

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR) # filtering out unnecessary warning from spacy

# pd.set_option('display.max_colwidth', -1)
# %%
"""
## Initial steps
The news data is obtained by running the [preprocessing notebook](https://github.com/prrao87/blog/tree/master/_notebooks/data/spacy_multiprocess) (`./data/preprocessing.ipynb`), which processes the raw text file downloaded from Kaggle and performs some basic cleaning on it. This step generates a file that contains the tabular data (stored as `nytimes.tsv`). A curated stopword file is also provided in [the same directory](https://github.com/prrao87/blog/tree/master/_notebooks/data/spacy_multiprocess).

Additionally, during initial testing, we can limit the size of the DataFrame being worked on (to a subset of the total number of articles) for faster execution. For the final run, disable the limit by setting it to zero.
"""

# %%
#collapse-hide
inputfile = "data/nytimes-sample.tsv"
stopwordfile = "stopwords/stopwords.txt"
limit = 0


# %%
"""
### Load spaCy model
Since we will not be doing any specialized tasks such as dependency parsing and named entity recognition in this exercise, these components are disabled when loading the spaCy model.
"""

# %%
"""
> Tip: spaCy has a `sentencizer` component that can be plugged into a blank pipeline.
"""

# %%
"""
The sentencizer pipeline simply performs tokenization and sentence boundary detection, following which lemmas can be extracted as token properties. 
"""

# %%
nlp = spacy.load('en_core_web_sm', disable=[ 'parser', 'ner'])
nlp.add_pipe("sentencizer")

# %%
"""
A method is defined to read in stopwords from a text file and convert it to a set in Python (for efficient lookup).
"""

# %%
def get_stopwords():
    "Return a set of stopwords read in from a file."
    with open(stopwordfile) as f:
        stopwords = []
        for line in f:
            stopwords.append(line.strip("\n"))
    # Convert to set for performance
    stopwords_set = set(stopwords)
    return stopwords_set

stopwords = get_stopwords()

# %%
"""
### Read in New York Times Dataset
The pre-processed version of the NYT news dataset is read in as a Pandas DataFrame. The columns are named `date`, `headline` and `content` - the text present in the content column is what will be preprocessed to remove stopwords and generate token lemmas.
"""

# %%
def read_data(inputfile):
    "Read in a tab-separated file with date, headline and news content"
    df = pd.read_csv(inputfile, sep='\t', header=None,
                     names=['date', 'headline', 'content'])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    return df

# %%
df = read_data(inputfile)
print(df.head(3))

# %%
"""
### Define text cleaner
Since the news article data comes from a raw HTML dump, it is very messy and contains a host of unnecessary symbols, social media handles, URLs and other artifacts. An easy way to clean it up is to use a regex that parses only alphanumeric strings and hyphens (so as to include hyphenated words) that are between a given length (3 and 50). This filters each document down to only meaningful text for the lemmatizer. 
"""

# %%
def cleaner(df):
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z0-9\-]{3,50}")
    df['clean'] = df['content'].str.findall(pattern).str.join(' ')
    if limit > 0:
        return df.iloc[:limit, :].copy()
    else:
        return df

# %%
df_preproc = cleaner(df)
df_preproc.head(3)

# %%
"""
Now that we have just the clean, alphanumeric tokens left over, these can be further cleaned up by removing stopwords before proceeding to lemmatization.
"""

# %%
"""
## Option 1: Sequentially process DataFrame column
The straightforward way to process this text is to use an existing method, in this case the `lemmatize` method shown below, and apply it to the `clean` column of the DataFrame using `pandas.Series.apply`. Lemmatization is done using the spaCy's underlying [`Doc` representation](https://spacy.io/usage/spacy-101#annotations) of each token, which contains a `lemma_` property. Stopwords are removed simultaneously with the lemmatization process, as each of these steps involves iterating through the same list of tokens.
"""

# %%
def lemmatize(text):
    """Perform lemmatization and stopword removal in the clean text
       Returns a list of lemmas
    """
    doc = nlp(text)
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords]
    return lemma_list

# %%
"""
The resulting lemmas are stored as a list in a separate column `preproc` as shown below.
"""

# %%
# %%time
# df_preproc['preproc'] = df_preproc['clean'].apply(lemmatize)
# df_preproc[['date', 'content', 'preproc']].head(3)

# %%
"""
Applying this method to the `clean` column of the DataFrame and timing it shows that it takes almost a minute to run on 8,800 news articles.
"""

# %%
"""
## Option 2: Use `nlp.pipe`
Can we do better? in the [spaCy documentation](https://spacy.io/api/language#pipe), it is stated that "processing texts as a stream is usually more efficient than processing them one-by-one". This is done by calling a language pipe, which internally divides the data into batches to reduce the number of pure-Python function calls. This means that the larger the data, the better the performance gain that can be obtained by `nlp.pipe`.

To use the language pipe to stream texts, a new lemmatizer method is defined that directly works on a spaCy `Doc` object. This method is then called in batches to work on a *sequence* of `Doc` objects that are streamed through the pipe as shown below.
"""

# %%
def lemmatize_pipe(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords] 
    return lemma_list

# def preprocess_pipe(texts):
#     preproc_pipe = []
#     for doc in nlp.pipe(texts, batch_size=20):
#         preproc_pipe.append(lemmatize_pipe(doc))
#     return preproc_pipe

# %%
"""
Just as before, a new column is created by passing data from the `clean` column of the existing DataFrame. Note that unlike in workflow #1 above, we do not use the `apply` method here - instead, the column of data (an iterable) is directly passed as an argument to the preprocessor pipe method.
"""

# %%
# %%time
# df_preproc['preproc_pipe'] = preprocess_pipe(df_preproc['clean'])
# df_preproc[['date', 'content', 'preproc_pipe']].head(3)

# %%
"""
Timing this workflow doesn't seem to show improvement over the previous workflow, but as per the spaCy documentation, one would expect that as we work on bigger and bigger datasets, this approach should show some timing improvement (on average).
"""

# %%
"""
## Option 3: Parallelize the work using joblib
We can do still better! The previous workflows sequentially worked through each news document to produce the lemma lists, which were then appended to the DataFrame as a new column. Because each row's output is completely independent of the other, this is an *embarassingly parallel* problem, making it ideal for using multiple cores.

The `joblib` library is recommended by spaCy for processing blocks of an NLP pipeline in parallel. Make sure that you `pip install joblib` before running the below section.

To parallelize the workflow, a few more helper methods must be defined. 

* **Chunking:** The news article content is a list of (long) strings where each document represents a single article's text. This data must be fed in "chunks" to each worker process started by joblib. Each call of the `chunker` method returns a generator that only contains that particular chunk's text as a list of strings. During lemmatization, each new chunk is retrieved based on the iterator index (with the previous chunks being "forgotten").


* **Flattening:** Once joblib creates a set of worker processes that work on each chunk, each worker returns a "list of lists" containing lemmas for each document. These lists are then combined by the executor to provide a 3-level nested final "list of lists of lists". To ensure that the length of the output from the executor is the same as the actual number of articles, a "flatten" method is defined to combine the result into a list of lists containing lemmas. As an example, two parallel executors would return a final nested list of the form: `[[[a, b, c], [d, e, f]], [[g, h, i], [j, k, l]]]`, where `[[a, b, c], [d, e, f]]` and `[[g, h, i], [j, k, l]]` refer to the output from each executor (the final output is then concatenated to a single list by joblib). A flattened version of this result would be `[[a, b, c], [d, e, f], [g, h, i], [j, k, l]]`, i.e. with one level of nesting removed.

In addition to the above methods, a similar `nlp.pipe` method is used as in workflow #2, on each chunk of texts. Each of these methods is wrapped into a `preprocess_parallel` method that defines the number of worker processes to be used (7 in this case), breaks the input data into chunks and returns a flattened result that can then be appended to the DataFrame. For machine with a higher number of physical cores, the number of worker processes can be increased further.

The parallelized workflow using joblib is shown below.
"""

# %%
from joblib import Parallel, delayed

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def process_chunk(texts):
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe

def preprocess_parallel(texts, chunksize=100):
    executor = Parallel(n_jobs=7, backend='multiprocessing', prefer="processes")
    do = delayed(process_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, len(df_preproc), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)

# %%
# %%time
df_preproc['preproc_parallel'] = preprocess_parallel(df_preproc['clean'], chunksize=1000)

# %%
df_out = df_preproc[['date', 'content', 'preproc_parallel']].head(3)
print(df_out)
