chunking_methods.md

```python
def tokenize_sentence_spacy(nlp, sent, lower_case=True, convert_num=False):
    tokens = [tok.text for tok in nlp(sent)]
    if lower_case:
        tokens = [t.lower() for t in tokens]
    if convert_num:
        tokens = ['<NUM>' if t.isdigit() else t for t in tokens]
    return tokens



def get_pos_nltk(doc, kg_name, lower_case=True, convert_num=True):
    doc_tagged=Counter()
    chunk_parser = get_pos_parser()

    for i, word in enumerate(tqdm(doc, total=len(doc))): 
        tokens = nltk.word_tokenize(word)
        tagged = nltk.pos_tag(tokens)
        cs = chunk_parser.parse(tagged)
        # print(cs)
        verb_phrases_raw = textacy.extract.matches(doc, pattern)


        tagged = nltk.pos_tag(tokens)[0][1]
        doc_tagged[tagged] +=1 

    total = sum(doc_tagged.values())
    doc_tagged_ratio = [(k,round(v/total, 4)) for k,v in doc_tagged.items()]
    doc_tagged_ratio = Counter(dict(doc_tagged_ratio))
    plot_pos(doc_tagged_ratio, f"analysis/{kg_name}_unigram_pos.png")
    print(doc_tagged)
    print(doc_tagged_ratio)
    return doc_tagged

def get_pos_parser():
    grammar = r"""
        NOUN: {<NN.*>+}          # Chunk sequences of DT, JJ, NN
        ADJECTIVE: {<JJ.*>+}               # Chunk prepositions followed by NP
        VERB: {<VB.*>+} # Chunk verbs and their arguments
        ADVERB: {<RB.*>+}
        PREPOSITION:{<IN>}
        DETERMINER: {<DT>}
        PRONOUN: {<PRP*>}
        """
    # grammar = 'NP: {<DT>?<JJ>*<NN>}'
    # grammar = '\n'.join([
	# 'NP: {<DT>*<NNP>}',
	# 'NP: {<JJ>*<NN>}',
	# 'NP: {<NNP>+}',
	# ])
    return nltk.RegexpParser(grammar)



def chunk_phrases_spacy(doc):
    '''
    # doc_inp = ('The talk will introduce reader about Use'
                    # ' cases of Natural Language Processing in'
                    # ' Fintech')
    # pattern = r'(<VERB>?<ADV>*<VERB>+)'
    
    # verb_phrases = textacy.extract.pos_regex_matches(doc, pattern)
    '''
    # pattern = [{'POS':'VERB'}]
    
    pattern = textacy.constants.POS_REGEX_PATTERNS["en"]["PP"]
    # pp_phrases = [("PP", chunk.text) for chunk in textacy.extract.matches(doc, pattern)]
    pp_phrases = [ ("PP", chunk.text) for chunk in textacy.extract.pos_regex_matches(doc, pattern)]
    # for chunk in pp_phrases:
        # print ("PP\t", chunk, doc)

    pattern = [{'POS': 'VERB', 'OP': '?'},
        {'POS': 'ADV', 'OP': '*'},
        {'POS': 'AUX', 'OP': '*'},
        {'POS': 'VERB', 'OP': '+'}]

    verb_phrases_raw = textacy.extract.matches(doc, pattern)
    verb_phrases = [("VP", chunk.text) for chunk in verb_phrases_raw]
    # Print all Verb Phrase
    # for chunk in verb_phrases:
        # print("VP\t", chunk)
    
    # Extract Noun Phrase to explain what nouns are involved
    pattern = textacy.constants.POS_REGEX_PATTERNS["en"]["NP"]
    noun_phrases = [ ("NP", chunk.text) for chunk in textacy.extract.pos_regex_matches(doc, pattern)]
    # noun_phrases= [("NP", chunk) for chunk in doc.noun_chunks]
    # for chunk in noun_phrases:
        # print ("NP\t", chunk)
 
    if len(pp_phrases)>0:
        return pp_phrases
    elif len(verb_phrases)>0:
        return verb_phrases
    elif len(noun_phrases)>0:
        return noun_phrases
    else:
        return [(token.pos_, token.text) for token in doc]
#    return (noun_phrases, verb_phrases, pp_phrases) 


# def get_chunk(sentence, chunk_parser):
    # sentence = "We saw the yellow dog"
    # tokens = nltk.word_tokenize(sentence)
    # print("tokens = {}".format(tokens))
    # tagged = nltk.pos_tag(tokens)

    # entities = nltk.chunk.ne_chunk(tagged)
    # print("entities = {}".format(entities))

    # grammar = r"""
        # NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and nouns
        # {<NNP>+}                # chunk sequences of proper nouns
    # """

    # cs = chunk_parser.parse(tagged)
    # cs = test_chunk(sentence)
    # chunk_output = Tree(cs)
    # for child in chunk_output:
        # if isinstance(child, Tree):               
            # if child.label() == 'NP':
                # for num in range(len(child)):
                    # if not (child[num][1]=='JJ' and child[num+1][1]=='JJ'):
                        # print (child[num][0])
    # return chunk_output 
    # return cs
    # print(cs)

def get_chunk_parser():
    grammar = r"""
        NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
        PP: {<IN><NP>}               # Chunk prepositions followed by NP
        VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
        CLAUSE: {<NP><VP>}           # Chunk NP, VP
        """
    # grammar = 'NP: {<DT>?<JJ>*<NN>}'
    # grammar = '\n'.join([
	# 'NP: {<DT>*<NNP>}',
	# 'NP: {<JJ>*<NN>}',
	# 'NP: {<NNP>+}',
	# ])
    return nltk.RegexpParser(grammar)

def get_chunk_spacy(concepts, kg_name, num_processes=1, debug=False):
    count_np=0
    chunk_counter = Counter()

    # with Pool(num_processes) as p:
        # for chunk, chunk_tag in tqdm(p.imap(constituents_parsing_bigram, concepts), total=len(concepts)): 
    for concept in tqdm(concepts, total=len(concepts)): 
        '''
        spacy_doc = textacy.make_spacy_doc(word, lang='en_core_web_sm')
        chunks = chunk_phrases_spacy(spacy_doc)
        # print(spacy_doc)
        # print("{}\t{}\t{}".format(i, word, chunks))
        '''
        chunk, chunk_tag = constituents_parsing_bigram(concept)
        chunk_counter[chunk_tag] +=1

    chunk_counter_norm = normalization_by_values(chunk_counter)
    plot_util(chunk_counter_norm, f"analysis/{kg_name}_bigram_phrasetag.png")
    print(chunk_counter)
    print(chunk_counter_norm)
    # print("The percentage of NP is {:.4f}".format(count_np/len(doc)))

````

1. load_triples() 
    head, relation, tail

2. get chunking tag for head and tail
    (head_tag, tail_tag)

3. statistics 
    dict{tuple:num}