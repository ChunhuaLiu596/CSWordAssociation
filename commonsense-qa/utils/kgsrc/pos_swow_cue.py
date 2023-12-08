# from pos_entities import load_concepts, get_pos_spacy 


# def main():
#     concepts_cn = load_concepts("data/swow/swow_cues.csv")
#     pos_cn, pos_cn_norm, phrase_cn = get_pos_spacy(concepts_cn, "swow_cues", concept_pos_path="data/swow/swow_cue_pos.csv", debug=False)
#     print(pos_cn)
#     print(pos_cn_norm)


# if __name__=='__main__':
#     main()

from pos_entities import load_concepts, get_pos_spacy, get_chunk_spacy
import pandas as pd
from tqdm import tqdm 
tqdm.pandas()

'''
usage:  python utils/kgsrc/pos_swow_cue.py 
'''
def split_cues_responsees():
    '''
    all_concepts = cues + responses 
    124626 =       12274 +  112352 
    '''
    path = "data/swow/conceptnet.en.csv"
    df = pd.read_csv(path, names=['rel', 'head', 'tail', 'weight'], sep='\t')
    df= df.query("rel=='forwardassociated'")

    df['head'] = df['head'].progress_apply(lambda x: str(x).replace("_", " "))
    df['tail'] = df['tail'].progress_apply(lambda x: str(x).replace("_", " "))

    cues = list(set(df['head'])) 
    responses = list(set(df['tail']))
    print(f"Cues: {len(cues)} Responses: {len(responses)}")

    return cues, responses  

def pos_cue(cues):
    '''
    cues: single word 12052
          multi-word 12052
    '''
    if cues is None:
        cues = load_concepts("data/swow/swow_cues.csv")
    concept_pos_path= "data/swow/swow_cue_pos.csv"
    concept_phrase_path = "data/swow/swow_cues_phrase.csv"
    type_count_path = 'data/swow/swow_cues_type.csv'

    pos, pos_norm, phrase = get_pos_spacy(cues, "swow_cues", concept_pos_path=concept_pos_path, debug=False)
    tag, tag_norm = get_chunk_spacy(phrase, "swow_cues", concept_phrase_path)
    # print(pos)
    # print(pos_norm)
    relative_frequency(concept_pos_path,concept_phrase_path, type_count_path, 'Cue' )


def pos_response(responses, cues, exclude_cues=True):
    if exclude_cues:
        cues_only = set(cues) - set(responses)
        responses = [r for r in responses if r not in cues_only]
        print(f"Exclue {len(cues_only)} words exist only in cues ")

    concept_pos_path="data/swow/swow_responses_pos.csv"
    concept_phrase_path = "data/swow/swow_responses_phrase.csv"
    type_count_path = 'data/swow/swow_responses_type.csv'

    pos, pos_norm, responses_phrase = get_pos_spacy(responses, "swow_responses",  concept_pos_path, debug=False)

    tag, tag_norm = get_chunk_spacy(responses_phrase, "swow_responses", concept_phrase_path)

    relative_frequency(concept_pos_path,concept_phrase_path, type_count_path , 'Response')


def relative_frequency(pos_path, phrase_type_path, type_count_path, concept_type):
    '''
    compute the relative frequency of each post/phrasal-type 
    '''
    # pos_path="data/swow/swow_responses_pos.csv"
    # phrase_type_path = "data/swow/swow_responsese_phrase.csv"

    df_pos = pd.read_csv(pos_path, names=[concept_type, 'tags'], sep='\t')
    df_pos['class'] = 'POS'
    # df_pos['tags'] = df_pos['tags'].apply(lambda x: " ".join(eval(f"{x}")))

    df_pt = pd.read_csv(phrase_type_path, names=[concept_type, 'tags', 'type_detail'], sep='\t')
    df_pt['class'] = 'Phrase'
    
    df = pd.concat([df_pos, df_pt])

    type_count=df.groupby('tags').count()
    type_count['ratio'] = type_count[concept_type] / type_count[concept_type].sum()
    type_count.sort_values('ratio', ascending=False)
    type_count.to_csv(type_count_path)

    print(type_count)
    print(f"save to {type_count_path}")



def post_processing(path=None, output_path=None):
    '''
    normalize frequency 
    format: rank, tags, cn, sw
    '''
    path = "./analysis/tag_node_pos_phrase_cn_sw.csv"
    df = pd.read_csv(path)
    df['CN_Ratio'] = df['CN']/df['CN'].sum()
    df['SW_Ratio'] = df['SW']/df['SW'].sum()

    output_path = "./analysis/tag_node_pos_phrase_cn_sw_ratio.csv"
    df['rank'] = df.index + 1
    df = df.drop(['Unnamed: 0'], axis=1).round(4)
    df = df[['rank', 'tags', 'CN', 'SW', 'CN_Ratio', 'SW_Ratio']]
    df.to_csv(output_path)
    print(f"save {output_path}")



def merge_cue_responses(cue_path, response_path, output_path):
    '''
    '''
    df1 = pd.read_csv(response_path).drop(['class', 'type_detail'], axis=1).round(4).sort_values('ratio', ascending=False)
    df2 = pd.read_csv(cue_path).drop(['class', 'type_detail'], axis=1).round(4).sort_values('ratio', ascending=False)

    df = df1.merge(df2, on='tags', suffixes=("_response", "_cue"))
    df['rank'] = df.index + 1 

    df.columns = ['tags', 'Response', 'Response_Ratio', 'Cue', 'Cue_Ratio', 'rank']

    df['Cue_Response'] = df['Cue'] + df['Response']
    df['Cue_Response_Ratio'] =  df['Cue_Response']/df['Cue_Response'].sum() 
    
    df = df[['rank', 'tags', 'Response', 'Response_Ratio', 'Cue', 'Cue_Ratio', 'Cue_Response', 'Cue_Response_Ratio']].round(4)
    df.to_csv(output_path, index=False)

    print(f"save {output_path}")


def main():
    # step1: tag each head and tail 
    cues, responses = split_cues_responsees()
    pos_cue(cues)
    pos_response(responses, cues)

    # step2: statistics
    concept_pos_path= "data/swow/swow_cue_pos.csv"
    concept_phrase_path = "data/swow/swow_cues_phrase.csv"
    type_count_path1 = 'data/swow/swow_cues_type.csv' 
    relative_frequency(concept_pos_path,concept_phrase_path, type_count_path1, "Cue" )

    concept_pos_path="data/swow/swow_responses_pos.csv"
    concept_phrase_path = "data/swow/swow_responses_phrase.csv"
    type_count_path2 = 'data/swow/swow_responses_type.csv' 
    relative_frequency(concept_pos_path,concept_phrase_path, type_count_path2, "Response")

    output_path = 'data/swow/swow_cue_response_type_ratio.csv'
    merge_cue_responses(type_count_path1, type_count_path2, output_path)
if __name__=='__main__':
    main()
