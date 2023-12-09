import spacy   #python -m spacy download en_core_web_sm
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
import nltk
from collections import Counter
import pandas as pd
# from spacy.en import English
# modified from: https://spacy.io/usage/linguistic-features

def customized_spacy_nlp(split_hyphens=False):
    '''
    modify tokenizer so that hyphenated words will not be splitted
    '''
    # default tokenizer
    nlp = spacy.load("en_core_web_sm")
    # doc = nlp("mother-in-law")
    # print([t.text for t in doc]) # ['mother', '-', 'in', '-', 'law']
    
    # modify tokenizer infix patterns
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),

            # if split_hyphen:
                # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            # else:
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer

    return nlp

pos_tags = [
    'NOUN',
    'PROPN', 
    'VERB',
    'ADJ',
    'ADV',
    'ADP',
    'AUX',
    'CCONJ', 
    'DET',
    'INTJ',
    'NUM',
    'PART',
    'PRON',
    'PUNCT',
    'SCONJ', 
    'SYM',
    'X',
    ]


phrase_tags = [
    'NP',
    'VP',
    'S',
    'PP',
    'ADJP',
    'X',
    'FRAG',
    'ADVP',
    'INTJ',
    'SBAR',
    'SQ',
    'SINV',
    'SBARQ',
    'UCP',
    'WHNP',
    'LST',
    'QP',
    'WHADVP',
    'RRC',
    'WHADJP',
    'WHPP',
    ]

pos_pos_tags = [[(x,y)] for x in pos_tags for y in pos_tags]
phrase_phrase_tags = [[(x,y)] for x in phrase_tags for y in phrase_tags]
pos_phrase_tags = list()
for x in pos_tags:
    for y in phrase_tags:
        pos_phrase_tags.append([(x,y)])
        pos_phrase_tags.append([(y,x)])

all_tags = pos_pos_tags + phrase_phrase_tags + pos_phrase_tags
# for x in phrase
# pos_phrase_tags = [ [(x,y)] for x in pos_tags for y in phrase_tags] + [ [(y,x)] for x in pos_tags for y in phrase_tags]

group_tags={x[0]:"PosPos" for x in pos_pos_tags}
group_tags.update({x[0]:"PhrasePhrase" for x in phrase_phrase_tags})
group_tags.update({x[0]:"PosPhrase" for x in pos_phrase_tags})

def phrasal_verb_recognizer(parsed) :
    '''
    parsed is a single sentence parsed by spacy 
    1)Verb + particle (particle verbs)
    2)Verb + particle + preposition (particle-prepositional verbs)
    '''
    is_phrasal_verb=False
    if len(parsed)==3:
        for token in parsed:
            if token.i==1 and token.dep_ == "prt" and token.head.pos_ == "VERB" and parsed[token.i+1].dep_ == "prep":
                verb = token.head.orth_
                particle = token.orth_
                preposition = parsed[token.i+1].orth_
                is_phrasal_verb=True
    if len(parsed)==2:
         for token in parsed:
            if token.i==1 and token.dep_ == "prt" and token.head.pos_ == "VERB":
                verb = token.head.orth_
                particle = token.orth_
                is_phrasal_verb=True
    return is_phrasal_verb


def compound_noun_recognizer(parsed) :
    '''
    input are concept nodes, like 'coffee pot'
    Returns:
        a bool variable specify whether the input is a compound noun

    parsed is a single sentence parsed by spacy 
    1) # NOUN + * (dep_ == 'compound') e.g., swimming pool
    2) * + NOUN,   e.g., washing machine
    '''
    is_compound_noun=False

    for token in parsed:
        if token.pos_ is 'NOUN': 
            if token.dep_ == 'compound' and token.i ==0: 
                is_compound_noun=True
                for j in token.ancestors:
                    print(token, j.text, j.pos_, j.dep_)
            else:  
                for j in token.lefts:
                    if j.dep_ == 'compound':
                        is_compound_noun=True
                        for k in token.ancestors:
                            print(token, k.text, k.pos_, k.dep_)

        # compounds = [c for c in compounds if c.i == 0 or doc[c.i - 1].dep_ != 'compound'] # Remove middle parts of compound nouns, but avoid index errors
    return is_compound_noun

# def post_process_tag_pairs(tags_path):
# selected_tags=['NOUN', 'PROPN', 'NP', 'VP', 'VERB', 'ADJ', 'ADV', 'S', 'PP', 'ADJP']
# selected_tag_pairs = [(x,y) for x in selected_tags for y in selected_tags]
# df1_data = [(k,1) for k in selected_tag_pairs]
# df_sel_tags=pd.DataFrame(Counter(dict(df1_data)).most_common(), columns=['sel_tags', 'idx'])


def content_word_recognizer(parsed):
    content_set = ('NOUN', 'VERB')
    output = set()
    for token in parsed:
        print(token, token.pos_)
        if token.pos_ in content_set: 
            output.add(token)
    return output 



def extract_compound_noun(doc):
    '''
    doc: nlp spacy parsed 
    Returns: 
       a list of compound nouns in the doc 
    '''
    compound_pairs = []
    for token in doc:
        if token.pos_ is 'NOUN':
            for j in token.lefts:
                if j.dep_ == 'compound':
                    compound_pairs.append(j.text+' '+token.text)
    return compound_pairs
