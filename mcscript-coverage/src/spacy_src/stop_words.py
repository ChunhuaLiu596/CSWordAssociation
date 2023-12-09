
__all__ = ['_STOP_WORDS', '_BLACK_LIST']

nltk_stopwords = set(['an', 'your', 'who', 'if', 'against', 'most', 'them', 'under', 'very', 'is', 'himself', 'being', 'have', 'or', 'such', 'am', 'be', 'and', 'o', 'themselves', "it's", "you're", "you'll", 'aren', 'between', 'all', 'yourself', 'ma', 'him', 'both', "needn't", 've', 'only', 'how', 'each', 'other', "haven't", 'some', 'my', 'd', 'i', 'from', 'couldn', 'own', 'no', 'her', 'again', 're', 'those', "mustn't", 'that', 'just', 'weren', "aren't", "hasn't", "shouldn't", 'yours', 'at', 'itself', "couldn't", 'which', 'me', "doesn't", 'off', "weren't", 'but', 'while', 'shan', 'its', 'into', 'before', "won't", 'their', 't', 'to', 'doesn', "that'll", 'this', "mightn't", "wasn't", 'over', 'through', 'the', 'can', "wouldn't", 'here', 'should', 'will', 'do', 'm', 'too', 'on', 'what', 'further', "didn't", 'because', 'shouldn', 'there', 'theirs', 'having', 'wasn', 'herself', 'y', 'needn', 'll', 'these', 'why', 'mightn', 'not', 'isn', 'it', "you've", 'didn', 'with', 'during', 'out', 'few', 'by', 'for', 's', 'of', 'he', 'nor', 'she', 'they', 'hadn', 'now', 'ain', 'you', 'than', 'we', 'myself', "shan't", 'don', 'below', 'hasn', 'as', 'had', 'up', "hadn't", 'ourselves', 'down', 'haven', "don't", 'doing', 'been', 'so', 'mustn', 'about', 'does', 'his', 'in', 'has', 'our', 'ours', 'are', 'yourselves', 'was', "she's", 'won', 'did', 'above', 'whom', 'then', "you'd", 'same', 'when', 'any', 'until', 'wouldn', 'were', 'after', 'once', "should've", 'hers', 'more', 'a', 'where', "isn't"])

spacy_stopwords = set(['anything', 'an', 'your', 'against', 'if', 'who', 'hence', 'most', 'them', 'upon', 'hereby', 'enough', 'under', 'is', 'very', 'really', 'top', 'himself', 'seeming', 'wherein', 'whole', 'twelve', 'being', 'have', 'name', 'or', 'am', 'and', 'be', 'eleven', 'made', 'such', 'themselves', 'thereby', 'go', 'all', 'between', 'latterly', 'nine', 'yourself', 'therefore', 'him', '’ve', 'both', 'hundred', 'others', 'towards', 'only', 'without', 'each', 'how', 'last', "'ve", 'six', 'five', 'other', 'some', 'whether', '’re', 'my', 'eight', 'from', 'i', 'nowhere', 'whereby', 'side', 'afterwards', 'no', 'often', 'her', 'own', 'again', 're', 'however', 'moreover', 'those', 'everywhere', 'just', 'another', 'that', 'toward', '‘ll', '’d', 'two', 'becoming', 'yours', 'could', 'put', 'someone', 'at', 'itself', 'which', 'mine', 'me', 'though', 'herein', 'off', 'but', 'while', 'hereafter', 'per', 'its', 'into', 'front', 'noone', 'before', 'thru', 'their', 'whither', 'to', 'due', 'next', 'behind', 'nevertheless', 'meanwhile', 'indeed', 'unless', 'this', 'ten', 'over', 'through', 'must', '’s', 'sixty', 'make', 'can', 'the', 'among', 'may', "'d", 'since', 'here', 'first', 'become', 'should', 'beyond', '’m', 'anyhow', 'do', 'ever', 'quite', 'using', 'whereafter', 'least', 'throughout', 'whereupon', 'will', 'across', 'too', 'on', 'except', 'whose', 'further', 'much', 'what', 'get', 'somehow', 'because', 'thus', 'many', 'although', 'there', 'well', 'herself', 'sometimes', 'take', 'seem', "'m", 'hereupon', 'beforehand', 'around', 'via', '‘d', 'regarding', 'these', 'why', 'seemed', 'call', 'not', 'anyway', 'everything', 'amongst', 'it', 'might', 'never', 'four', 'onto', 'whenever', 'amount', 'anywhere', 'nobody', '‘s', 'one', 'during', 'few', 'out', 'latter', 'by', 'thence', 'with', 'still', 'former', 'for', 'neither', 'of', 'he', 'part', 'somewhere', 'nor', 'ca', 'please', 'she', 'they', 'whatever', 'empty', 'whence', 'show', 'now', 'serious', 'else', 'fifty', 'n’t', "n't", 'you', 'than', 'we', 'everyone', 'myself', '’ll', 'see', 'third', 'none', 'below', 'us', 'various', 'together', 'namely', 'as', 'something', 'had', 'perhaps', 'formerly', 'rather', 'whoever', 'elsewhere', 'therein', 'up', 'down', 'keep', 'becomes', 'ourselves', '‘re', 'almost', 'doing', 'every', 'say', 'anyone', 'sometime', 'would', 'twenty', 'been', 'along', 'done', 'already', 'otherwise', 'cannot', 'alone', 'so', 'about', 'does', 'bottom', 'his', 'in', 'thereupon', 'wherever', 'has', '‘ve', 'beside', 'our', 'seems', 'within', 'are', 'ours', 'used', 'n‘t', 'was', 'yourselves', "'s", 'move', 'also', 'give', 'did', 'above', '‘m', 'always', 'less', 'whom', 'thereafter', 'nothing', 'same', 'then', "'re", 'when', 'even', 'any', 'besides', 'either', 'until', 'yet', "'ll", 'fifteen', 'mostly', 'were', 'whereas', 'after', 'forty', 'several', 'became', 'full', 'once', 'hers', 'more', 'a', 'three', 'back', 'where'])

# from http://armandbrahaj.blog.al/2009/04/14/list-of-english-stop-words/
extra_stopwords={'really', 'o', "it's", "you're", "you'll", 'aren', 'ma', '’ve', "'ve", "needn't", 've', "haven't", '’re', 'd', 'couldn', "mustn't", 'just', 'weren', '‘ll', '’d', "aren't", "hasn't", "shouldn't", "couldn't", "doesn't", "weren't", 'shan', "won't", 't', 'unless', 'doesn', "that'll", "mightn't", "wasn't", '’s', 'make', "wouldn't", "'d", 'quite', 'using', 'm', "didn't", 'having', 'theirs', 'wasn', "'m", 'y', '‘d', 'needn', 'll', 'regarding', 'mightn', 'isn', "you've", 'didn', '‘s', "'s", 's', 'ca', 'hadn', 'fifty', 'n’t', 'ain', "n't", '’ll', "shan't", 'don', 'various', 'hasn', "hadn't", 'haven', "don't", '‘re', 'say', 'doing', 'mustn', 'does', 'n‘t', 'used', "she's", 'won', 'did', '‘m', "'re", "you'd", '’m', 'wouldn', "'ll", "should've", '‘ve', 'shouldn', "isn't"}


_STOP_WORDS = set().union(*[nltk_stopwords, spacy_stopwords, extra_stopwords])


# the lemma of it/them/mine/.. is -PRON-
_BLACK_LIST = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be", "way", "say"
                 ])


pronoun_list = set(["my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our", "we"])



'''
help:
import nltk
# nltk.download('stopwords', quiet=True)
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))
import spacy
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words

'''