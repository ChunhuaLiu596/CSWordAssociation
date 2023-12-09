import sys
from tqdm import tqdm 
from datasets import load_dataset
import nltk
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import pandas as pd
from datetime import datetime

from nltk import word_tokenize
from nltk.collocations import BigramCollocationFinder
tqdm.pandas()

def load_gigaword_dataset():
    # downloading and preparing dataset gigaword/default (download: 551.61 MiB, generated: 918.35 MiB, post-processed: Unknown size, total: 1.44 GiB) to /home/chunhua/.cache/huggingface/datasets/gigaword/default/1.2.0/ea83a8b819190acac5f2dae011fad51dccf269a0604ec5dd24795b64efb424b6...
    dataset = load_dataset("gigaword")
    # for i in range(10):
    # print(dataset['train'][i])
    print(len(dataset['train']))
    return dataset

def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

dataset = load_gigaword_dataset()
df = pd.DataFrame(dataset['train'][:]) #3,803,957 documents, about 4M documents, 873M .arrow

debug=False
# debug=True
if debug:
    N = 10
    df = df.head(N)
print(df)
print("basic clean ...")

t1 = datetime.now()
print("Might take a long time.")
print(f"Time: {t1}")

words = basic_clean(''.join(str(df['document'].tolist())))

def get_ngrams_count(words, n):
    output_path = f'output/gigawords_{n}gram.csv'

    t1 = datetime.now()
    print("Might take a long time.")
    print(f"Time: {t1}")

    ngrams = pd.Series(nltk.ngrams(words, n)).value_counts()

    df_ngrams = pd.DataFrame(data = {"bigram": ngrams.index, "count": ngrams.values})
    df_ngrams["bigram"] = df_ngrams["bigram"].progress_apply(lambda x: ' '.join(x))
    df_ngrams.to_csv(output_path, index=False)

    t2 = datetime.now()
    print(f"Cost {t2 - t1} ")
    print(f"save {output_path} {len(df_ngrams.index) } lines")
    # print(df_bigrams)

get_ngrams_count(words, int(sys.argv[1]))
# get_ngrams_count(words,1)
# get_ngrams_count(words,2)
# get_ngrams_count(words,3)
# get_ngrams_count(words,4)
# get_ngrams_count(words,5)
#get_ngrams_count(words, sys.argv[1])
#
# text = "obama says that obama says that the war is happening"
# finder = BigramCollocationFinder.from_words(word_tokenize(dataset['train'][0]['document']))

# for k,v in finder.ngram_fd.items():
    # print(k,v)
