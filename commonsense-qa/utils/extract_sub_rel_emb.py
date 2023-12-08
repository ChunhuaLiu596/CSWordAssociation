import numpy as np
import argparse

# try:
    # from .conceptnet import merged_relations, merged_relations_7rel, load_merge_relation, relation_groups, relation_groups_7rel
# except ImportError:
    # from conceptnet import merged_relations, merged_relations_7rel, load_merge_relation, relation_groups, relation_groups_7rel

def get_merged_relations(kg_name):
    if kg_name in ('cpnet'):
        from conceptnet import merged_relations
    elif kg_name in ('cpnet7rel'):
        from conceptnet import merged_relations_7rel as merged_relations
    elif kg_name in ('cpnet1rel'):
        from conceptnet import merged_relations_1rel as merged_relations
    elif kg_name in ('swow'):
        from swow import merged_relations
    elif kg_name in ('swow1rel'):
        from swow import merged_relations_1rel as merged_relations

    return merged_relations

def load_resources(cpnet_vocab_path):
    global concept2id, id2concept, relation2id, id2relation
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]

# def extract_rel_embedding_cpnet7rel(output_path):

#     cpnet_17rel_emb = np.load('./data/transe/glove.transe.sgd.rel.npy')
#     rel_embeddings = { rel: emb for rel,emb in zip(merged_relations, cpnet_17rel_emb)}
    
#     cpnet_17rel = merged_relations_7rel
#     rel_embeddings_sub = np.zeros((len(merged_relations_7rel), cpnet_17rel_emb.shape[1]), dtype="float32")

#     for i, rel in enumerate(merged_relations_7rel):
#         rel_embeddings_sub[i]= rel_embeddings[rel]
#     np.save(output_path, rel_embeddings_sub)

#     print('save {1} relation embeddings to {0}'.format(output_path, len(rel_embeddings_sub)))
#     print('done')


def extract_sub_rel_embedding(input_kg_name, input_emb_path, outout_kg_name, output_emb_path):

    merged_relations =  get_merged_relations(input_kg_name) 
    print("merged_relations:{}".format(merged_relations))
    merged_relations_nrel = get_merged_relations(outout_kg_name)
    print("merged_relations_nrel:{}".format(merged_relations_nrel))

    merged_rel_emb = np.load(input_emb_path)
    rel2emb = { rel: emb for rel,emb in zip(merged_relations, merged_rel_emb)}
    
    sub_rel_emb = np.zeros((len(merged_relations_nrel), merged_rel_emb.shape[1]), dtype="float32")

    for i, rel in enumerate(merged_relations_nrel):
        sub_rel_emb[i]= rel2emb[rel]
    np.save(output_emb_path, sub_rel_emb)

    print('save {1} relation embeddings to {0}'.format(output_emb_path, len(sub_rel_emb)))
    print('done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_kg_name", type=str, choices=['cpnet', 'swow'])
    parser.add_argument("--output_kg_name", type=str, choices=['cpnet7rel', 'cpnet1rel','swow1rel'])
    parser.add_argument("--input_emb_path", type=str, help="input relation embedding path")
    parser.add_argument("--output_emb_path", type=str, help="output relation embedding path")
    args = parser.parse_args()
    input_emb_path = f'data/cpnet7rel/glove.transe.sgd.rel.npy'
    #output_path='data/cpnet7rel/glove.transe.sgd.rel.npy'
    # extract_rel_embedding_cpnet7rel(output_path)

    # output_path='data/cpnet7rel/glove.transe.sgd.rel.npy'
    extract_sub_rel_embedding(args.input_kg_name, args.input_emb_path, args.output_kg_name, args.output_emb_path)