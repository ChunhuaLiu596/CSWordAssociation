import os, sys
import pandas as pd
import spacy 
from tqdm import tqdm
tqdm.pandas()


def format_normalize(path, output_path):
    '''
    convert a (r,h,t) triple format into a (r,h,t,w) format
    '''
    df = pd.read_csv(path)
    
    df['weight'] = [1.0]*len(df.index)
    
    df.to_csv(output_path, sep='\t', header=False, index=False)
    print(f"save {output_path} {len(df.index)} lines")

nlp = spacy.load("en_core_web_sm")

# all_stopwords = nlp.Defaults.stop_words
# verbs_srl_retain = ['put', 'make', 'become', 'say', 'made', 'go', 'used', 'becoming', 'give', 'show','becomes'] 
# for word in verbs_srl_retain:
    # all_stopwords.remove(word) #this influence the verbs when selecting triples from SRL 


def content_word_recognizer_old(doc, nlp, verbose=False):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    content_set = ('NOUN', 'VERB')
    # print("doc", doc)
    doc = [word for word in doc.split() if not word in all_stopwords]
    doc = " ".join(doc)

    output = set()
    output_lemma = set()
    parsed = nlp(doc)
    
    for token in parsed:
        # print(token, token.pos_)
        if token.pos_ in content_set: 
            output.add(token.text)
            output_lemma.add(token.lemma_)

        if token.pos_ is 'NOUN':
            # print("noun", token.text)
            for j in token.lefts:
                # print(j.dep_)
                if j.dep_  in ('compound', 'amod'):
                    output.add(j.text+' '+token.text)
                    output_lemma.add(j.text+' '+token.lemma_)
                    # output_lemma.add(j.lemma_+' '+token.lemma_)
                    print(j.dep_, j.lemma_+' '+token.lemma_)
                else:
                    output.add(token.text)
                    output_lemma.add(token.lemma_)

    return output, output_lemma



def content_word_recognizer(doc, nlp, verbose=False):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    content_set = ('NOUN', 'VERB')
    lemma_tags = {"NNS", "NNPS"}
    # print("doc", doc)

    # doc = [word for word in doc.split() if not word in all_stopwords]
    # doc = " ".join(doc)

    output = set()
    output_lemma = set()
    parsed = nlp(doc)
    
    for token in parsed:
        iscompound=False
        compound_left = ""
        if token.pos_ is 'NOUN':

            for j in token.lefts:
                if j.dep_  in ('compound', 'amod'):
                    output.add(j.text+' '+token.text)
                    compound = j.text+' '+token.lemma_ 
                    output_lemma.add(j.text+' '+token.lemma_)
                    if verbose:
                        print(j.dep_, j.lemma_+' '+token.lemma_)
                    iscompound=True
                    compound_left = j.text

            if not iscompound:
                if verbose:
                    print("noun", token.text)
                output.add(token.text)
                output_lemma.add(token.lemma_)

    if compound_left in output_lemma:
        output_lemma.remove(compound_left)
        output.remove(compound_left)

    if len(output)==0:
        for token in parsed:
            # print(token, token.pos_)
            if token.pos_ in content_set: 
                if token.text in all_stopwords:
                    continue 
                output.add(token.text)
                output_lemma.add(token.lemma_)
    if verbose:
        print(output, output_lemma)
    return output, output_lemma

all_stopwords= ('anyway', 'or', 'to', 'before', 'quite', 'nobody', 'just', 'although', 'some', 'least', 'thereafter', 'two', 'until', 'n’t', 'nine', 'many', 'same', 'your', 'out', 'otherwise', 'top','how', 'no', 'nor', 'twenty', 'they', 'off', 'down', 'noone', 'being', 'must', 'never', 'cannot', 'empty', 'eleven', 'myself', 'neither', 'a', 'rather', 'therefore', 'then', 'we', 'into', 'ourselves', 'please', 'indeed', 'beyond', 'among', 'front', 'up', 'other', 'through', 'whoever', 'each', 'behind', 'once', 'hereafter', 'herself', 'because', 'when', 'another', 'first', 'thru', 'was', 'did','somewhere', 'doing', 'be', 'part', 'mostly', 'my', 'next', 'becoming', 'in', 'forty', 'you', 'amount', 'eight', 'here', 'himself', 'besides', 'itself', 'an', 'nevertheless', 'thereby', 'upon', 'over', 'anyhow', 'along', 'someone', 'without', 'its', 'back', 'somehow', 'whereby', 'might', '’d', 'but', 'again', 'none', 'whether', '’ll', 'something', 'one', 'any', 'others', 'beforehand', 'who', 'full', 'him', 'serious', 'below', 'why', 'meanwhile', 'may', 'ours', 'own', 'ten', 'anything', 'their', 'i', 'of', 'his', 'most', 'not', 'unless', 'more', 'n‘t', 'everywhere', 'with', 'about', 'it', 'towards', 'whenever', 'since', 'where', '’ve', 'few', 'herein', 'should', 'yourselves', 'hereby', 'as', 'seem', 'if', "'re", 'is', 'whither', '‘s', 'could', 'however', 'seeming', '’s', 'ca', 'were', 'several', 'there', 'whereupon', 'perhaps', 'three', '‘d', '‘ve', 'do', 'very', 'last', 'everyone', 'after', 'been', 'show', 'whereafter', 'together', 'hereupon', 'take', 'twelve', 'yet', 'too', 'either', 'less', 'themselves', 'name', "'s", 'what', 'during', 'hundred', 'call', 'against', 'various', 'would', 'have', 'always', 'every', 'at', 'except', 'formerly', 'per', 'seems', 'whence', "'ve", 'moreover', 'onto', 'he', 'throughout', 'yours', 'whole', 'yourself', 'thereupon', 'therein', 'now', 'mine', 'fifty', 'on', 'ever', "n't", "'m", 'such', 'done', 'latterly', '’re', 'them', 'whereas', "'d", 'her', 'under', "'ll", 'from', 'hence', 'regarding', 'us', 'sixty', 'thence', 'wherein', 'much', 'see', 'the', 'sometime', 'else', 'former', 'wherever', 'alone', 'am', 'everything', 'which', '‘m', 'that', 'four', 'can', 'had', '‘ll', '’m', 'afterwards', 'are', 'become', 'fifteen', 'within', 'sometimes', 'those', 'five', 'side', 'due', 'for', 'still', 'has', 'so', 'six', 'around', 'this', 'above', 'while', 'all', 'third', 'does', 'already', 'across', 'elsewhere', 'namely', 'via', 'anyone', 'toward', 'hers', 'nothing', 'than', 'our', 'though', 'well', 'she', 'these', 'beside', '‘re', 'really', 'even', 'only', 'further', 'latter', 'often', 'put', 're', 'whom', 'by', 'me', 'say', 'also', 'thus', 'and', 'almost', 'anywhere', 'between', 'both', 'bottom', 'seemed', 'whose', 'nowhere', 'will', 'amongst', 'whatever')

def content_word_recognizer_np(doc, nlp=None, verbose=False):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    # content_set = ('NOUN', 'VERB')
    content_set = ('VERB')
    lemma_tags = {"NNS", "NNPS"}

    # 'on a hike in the woods' -> 'hike' 'wood'
    #  'our sleeping bags' -> sleeping bag 
    print("doc", doc)
    # doc = [word for word in doc.split() if word not in all_stopwords]
    # doc = ' '.join(doc)
    # word  = ' '.join([word for word in doc.split() if word not in all_stopwords])

    if verbose:
        print(f"concept: {doc}")
    output = set()
    output_lemma = set()
    parsed = nlp(doc)

    for chunk in parsed.noun_chunks:
        if chunk.text in all_stopwords:
            continue 

        output.add(chunk.text)
        print('NP', chunk.text)
        lemmas = []
        for token in chunk:
            if token.text in all_stopwords:
                continue 
            if token.tag_ in lemma_tags:
                lemma = token.lemma_
                lemmas.append(lemma)
            else:
                lemmas.append(token.text)
        # output_lemma.add(chunk.text)
        output_lemma.add(" ".join(lemmas))

    if len(output)==0:
        for token in parsed:
            if token.pos_ in content_set: 
                if token.text in all_stopwords:
                    continue 
                output.add(token.text)
                output_lemma.add(token.lemma_)
            # if token.pos_ is 'NOUN':
            #     for j in token.lefts:
            #         if j.dep_ == 'compound':
            #             output.add(j.text+' '+token.text)
            #             output_lemma.add(j.lemma_+' '+token.lemma_)
    if verbose:
        print("content_word_recognizer " , doc, output, output_lemma)
    return output, output_lemma


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


def extract_compound_noun(parsed):
    '''
    doc: nlp spacy parsed 
    Returns: 
       a list of compound nouns in the doc 
    '''
    compound_pairs = []
    for token in parsed:
        if token.pos_ is 'NOUN':
            for j in token.lefts:
                if j.dep_ == 'compound':
                    compound_pairs.append(j.text+' '+token.text)
    return compound_pairs


def reconstruct_triple(rel, head, tail, nlp=None, verbose=False):
    # nlp = spacy.load("en_core_web_sm")
    if verbose:
        print(rel,head, tail)
    head_set, head_set_lemma = content_word_recognizer(head, nlp, verbose)
    tail_set, tail_set_lemma = content_word_recognizer(tail, nlp, verbose)

    if verbose: 
        print("head", head_set, head_set_lemma)
        print("tail", tail_set, tail_set_lemma)

    triple = set()
    for h in head_set:
        for t in tail_set: 
            triple.add((rel,h,t))
            if verbose:
                print(rel,h,t)

    triple_lemma = set() 
    for h in head_set_lemma:
        for t in tail_set_lemma: 
            triple_lemma.add((rel,h,t))
    return triple, triple_lemma


def node_normalize(path, output_path):
    df = pd.read_csv(path, sep='\t', names=['rel', 'head', 'tail', 'weight'])
    print(df.head(5))
    df['head_set'] = df['head'].progress_apply(lambda x: content_word_recognizer(x))
    df['tail_set'] = df['tail'].progress_apply(lambda x: content_word_recognizer(x))

    # df['triple'] = df[['head_set','tail_set','rel']].progress_apply(lambda x: reconstruct_triple(x))
    triples = dict()
    for hs, ts, r in zip(df['head_set'], df['tail_set'], df['rel']):
        for h in hs:
            for t in ts: 
                if not h in triples:
                    triples[h] = {}
                triples[h][t] = r
    write_dict(triples, output_path)


def node_normalize(path, output_path):
    df = pd.read_csv(path, sep='\t', names=['rel', 'head', 'tail', 'weight'])

    columns_to_keep = ['rel', 'head', 'tail', 'triple']
    df.to_csv(output_path, index=False)
    print(f"save {output_path} {len(df.index)} lines ")

def write_dict(triples, output_path):
    weight = 1.0
    count = 0
    with open(output_path, 'w') as fout:
        for h, ts in triples.items():
            for t,r in ts.items():
                string = f'{r}\t{h}\t{t}\t{weight}\n'
                fout.write(string)
                count+=1
    print(f"save {output_path} {count} lines")
    # df = pd.DataFrame(list(triples), columns =['rel', 'head', 'tail', 'weight'])
    # df = df.drop_duplicates()
    # df['weight'] = [1.0]*len(df.index)
    # df.to_csv(output_path, index=False, header=False, sep='\t')
    # print(f"save {output_path} {len(df.index)} lines")


def main(suffix):
    path = f'output/{suffix}_frame.csv'
    output_path_mid = f'output/{suffix}.en.csv'
    output_path_final = f'output/{suffix}.raw.en.csv'
    format_normalize(path, output_path_mid)
    node_normalize(output_path_mid, output_path_final)

if __name__=='__main__':
    # suffix = sys.argv[1]
    # main(suffix)
    triple = ('ARG1V', 'our sleeping bags', 'put')
    # triple = ('ARG1V', 'some smores', 'made')
    # triple = ('ARG1V', 'on a hike in the woods', 'went')
    triple = ('ARG01', 'we', 'our sleeping bags')
    triples = "{('ARG01', 'we', 'all of the food that we would need to have to eat'), ('ARG1V', 'we', 'got'), ('ARG0V', 'we', 'went'), ('ARG1V', 'our sleeping bags', 'put'), ('ARG1V', 'that', 'eat'), ('ARG01', 'we', 'some smores'), ('ARG1V', 'on a hike in the woods', 'went'), ('ARG1V', 'camping', 'went'), ('ARG1V', 'we', 'became'), ('ARG0V', 'we', 'made'), ('ARG0V', 'we', 'walked'), ('ARG0V', 'we', 'told'), ('ARG0V', 'we', 'eat'), ('ARG1V', 'another fire', 'set'), ('ARG1V', 'scary stories', 'told'), ('ARG1V', 'it', 'getting'), ('ARG0V', 'we', 'packed'), ('ARG0V', 'we', 'go'), ('ARG1V', 'all of the food that we would need to have to eat', 'packed'), ('ARG01', 'we', 'some marsh mellows'), ('ARG1V', 'we all', 'went'), ('ARG1V', 'to have to eat', 'need'), ('ARG01', 'we', 'that'), ('ARG01', 'we', 'granola bars'), ('ARG01', 'we', 'on a hike in the woods'), ('ARG01', 'we', 'to go camping'), ('ARG0V', 'we', 'ate'), ('ARG0V', 'we', 'put'), ('ARG0V', 'we', 'drove'), ('ARG1V', 'we', 'getting'), ('ARG1V', 'our tent', 'set'), ('ARG0V', 'we', 'decided'), ('ARG01', 'we', 'camping'), ('ARG0V', 'bags', 'sleeping'), ('ARG1V', 'camping', 'go'), ('ARG1V', 'some smores', 'made'), ('ARG0V', 'we', 'got'), ('ARG0V', 'we all', 'sleep'), ('ARG01', 'we', 'another fire'), ('ARG01', 'we', 'our food'), ('ARG1V', 'our food', 'ate'), ('ARG1V', 'everything', 'packed'), ('ARG0V', 'we', 'set'), ('ARG01', 'we', 'our tent'), ('ARG0V', 'we', 'roasted'), ('ARG1V', 'some marsh mellows', 'roasted'), ('ARG0V', 'we', 'need'), ('ARG01', 'we', 'to have to eat'), ('ARG01', 'we', 'our sleeping bags'), ('ARG01', 'we', 'scary stories'), ('ARG1V', 'to go camping', 'decided'), ('ARG01', 'we', 'everything'), ('ARG1V', 'granola bars', 'packed')}"
    triples ="{('ARG1V', 'the flowers', 'been'), ('ARG0V', 'i', 'removed'), ('ARG1V', 'to grow', 'continued'), ('ARG1V', 'they', 'grow'), ('ARG0V', 'i', 'watered'), ('ARG1V', 'sure to water it every day', 'made'), ('ARG1V', 'the plants', 'grow'), ('ARG1V', 'bees', 'arrived'), ('ARG1V', 'a few seeds', 'placed'), ('ARG1V', 'to grow where the flowers had been', 'started'), ('ARG0V', 'i', 'placed'), ('ARG0V', 'i', 'went'), ('ARG1V', 'any weeds that were growing in the garden', 'removed'), ('ARG0V', 'i', 'watched'), ('ARG01', 'i', 'a few seeds'), ('ARG0V', 'i', 'made'), ('ARG1V', 'to grow until they were big enough to pick', 'continued'), ('ARG1V', 'small holes', 'dug'), ('ARG01', 'i', 'any weeds that were growing in the garden'), ('ARG1V', 'the garden', 'watered'), ('ARG01', 'i', 'seeds , fertilizer , and gardening tools'), ('ARG01', 'i', 'it'), ('ARG0V', 'i', 'dug'), ('ARG1V', 'they', 'were'), ('ARG1V', 'seeds , fertilizer , and gardening tools', 'purchase'), ('ARG1V', 'flowers', 'appeared'), ('ARG1V', 'the holes', 'covered'), ('ARG01', 'i', 'the holes'), ('ARG0V', 'i', 'water'), ('ARG1V', 'the flowers', 'pollinate'), ('ARG0V', 'bees', 'pollinate'), ('ARG01', 'i', 'sure to water it every day'), ('ARG1V', 'to ripen', 'began'), ('ARG01', 'i', 'small holes'), ('ARG1V', 'it', 'water'), ('ARG01', 'bees', 'the flowers'), ('ARG1V', 'they', 'ripen'), ('ARG1V', 'vegetables', 'grow'), ('ARG0V', 'i', 'purchase'), ('ARG01', 'i', 'the garden'), ('ARG1V', 'any weeds', 'growing'), ('ARG0V', 'i', 'covered')}"
    triples = "{('ARG1V', 'the holes', 'covered'), ('ARG0V', 'bees', 'pollinate'), ('ARG1V', 'to grow until they were big enough to pick', 'continued'), ('ARG01', 'i', 'the holes'), ('ARG1V', 'they', 'ripen'), ('ARG01', 'i', 'sure to water it every day'), ('ARG1V', 'the plants', 'grow'), ('ARG0V', 'i', 'covered'), ('ARG0V', 'i', 'removed'), ('ARG0V', 'i', 'made'), ('ARG1V', 'a few seeds', 'placed'), ('ARG1V', 'seeds , fertilizer , and gardening tools', 'purchase'), ('ARG01', 'bees', 'the flowers'), ('ARG1V', 'the flowers', 'been'), ('ARG1V', 'it', 'water'), ('ARG1V', 'they', 'were'), ('ARG0V', 'i', 'watered'), ('ARG1V', 'the flowers', 'pollinate'), ('ARG1V', 'the garden', 'watered'), ('ARG01', 'i', 'small holes'), ('ARG01', 'i', 'the garden'), ('ARG1V', 'to grow', 'continued'), ('ARG1V', 'to grow where the flowers had been', 'started'), ('ARG1V', 'vegetables', 'grow'), ('ARG1V', 'they', 'grow'), ('ARG0V', 'i', 'water'), ('ARG0V', 'i', 'dug'), ('ARG01', 'i', 'a few seeds'), ('ARG1V', 'any weeds', 'growing'), ('ARG1V', 'sure to water it every day', 'made'), ('ARG01', 'i', 'it'), ('ARG0V', 'i', 'placed'), ('ARG0V', 'i', 'watched'), ('ARG01', 'i', 'seeds , fertilizer , and gardening tools'), ('ARG01', 'i', 'any weeds that were growing in the garden'), ('ARG1V', 'bees', 'arrived'), ('ARG1V', 'small holes', 'dug'), ('ARG0V', 'i', 'purchase'), ('ARG1V', 'any weeds that were growing in the garden', 'removed'), ('ARG0V', 'i', 'went'), ('ARG1V', 'to ripen', 'began'), ('ARG1V', 'flowers', 'appeared')}"
    triples = "{('ARG1V', 'seeds , fertilizer , and gardening tools', 'purchase')}"
    triples = "{('ARG1V', 'any weeds that were growing in the garden', 'removed')}"
    triples = "{('ARG1V', 'sure to water it every day', 'made')}"
    verbose =True
    # verbose = False 
    for triple in eval(triples):
        print("# ", triple)
        t, tl = reconstruct_triple(triple[0], triple[1], triple[2], nlp=None, verbose=verbose)
        print("$ ", t, tl) 
        print(" ")