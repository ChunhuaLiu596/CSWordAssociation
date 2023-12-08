import os
import json
import numpy as np
from tqdm import tqdm
import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('kg_name', default='conceptnet', help='specify the knowledg graph')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read("paths.cfg")

PAD_TOKEN="_PAD"

class PredictRelations(object):
    def __init__(self,):
        self.use_swow_glove_emb = False
        self.use_swow_transe_emb = False
        self.use_swow19_transe_emb = False
        self.use_swow19_glove_emb = True

        self.load_vocabs()
        ent_embs, rel_embs = self.create_pretrained_concept_embs()
        triples_text = self.predict_relations(ent_embs, rel_embs)
        self.write_files(triples_text, config["paths_{}".format(args.kg_name)]["triples_output"], config["paths_{}".format(args.kg_name)]["concept_vocab"], config["paths_{}".format(args.kg_name)]["relation_vocab"])


    def load_vocabs(self):
        self.cpnet_ent2id, self.cpnet_ent_embs = load_embs_from_npy(config["paths_cpnet"]["concept_vec_npy_transe"], config["paths_cpnet"]["concept_vocab"] )
        self.cpnet_rel2id, self.cpnet_rel_embs = load_embs_from_npy(config["paths_cpnet"]["relation_vec_npy_transe"], config["paths_cpnet"]["relation_vocab"] )

        self.cpnet_id2rel= {idx:rel for rel, idx in self.cpnet_rel2id.items()}

        if self.use_swow19_transe_emb:
            self.swow_ent2id, self.swow_ent_embs = load_embs_from_npy(config["paths_swow_19relations"]["concept_vec_npy_transe"], config["paths_swow_19relations"]["concept_vocab"] )
            self.swow_rel2id, self.swow_rel_embs = load_embs_from_npy(config["paths_swow_19relations"]["relation_vec_npy_transe"], config["paths_swow_19relations"]["relation_vocab"] )

        if self.use_swow_transe_emb:
            self.swow_ent2id, self.swow_ent_embs = load_embs_from_npy(config["paths_swow_2relations"]["concept_vec_npy_transe"], config["paths_swow_2relations"]["concept_vocab"] )
            self.swow_rel2id, self.swow_rel_embs = load_embs_from_npy(config["paths_swow_2relations"]["relation_vec_npy_transe"], config["paths_swow_2relations"]["relation_vocab"] )

        if self.use_swow_glove_emb:
            self.swow_ent2id, self.swow_ent_embs = load_embs_from_npy(config["paths_swow_2relations"]["concept_vec_npy_glove"]+'.max.npy', config["paths_swow_2relations"]["concept_vocab"] )
            self.swow_rel2id, self.swow_rel_embs = load_embs_from_npy(config["paths_swow_2relations"]["relation_vec_npy_glove"]+'.max.npy', config["paths_swow_2relations"]["relation_vocab"] )


        if self.use_swow19_glove_emb:
            self.swow_ent2id, self.swow_ent_embs = load_embs_from_npy(config["paths_swow_19relations"]["concept_vec_npy_glove"]+'.max.npy', config["paths_swow_19relations"]["concept_vocab"] )
            self.swow_rel2id, self.swow_rel_embs = load_embs_from_npy(config["paths_swow_19relations"]["relation_vec_npy_glove"]+'.max.npy', config["paths_swow_19relations"]["relation_vocab"] )

        self.swow_id2rel= {idx:rel for rel, idx in self.swow_rel2id.items()}
        self.swow_id2ent= {idx:ent for ent, idx in self.swow_ent2id.items()}


    def create_pretrained_concept_embs(self, dim=100):

        embeddings = {}
        n = len(self.swow_ent2id.keys())
        overlap_concept = set()
        for concept in tqdm(self.swow_ent2id.keys(), total=n, desc='converting embeddings'):
            if concept not in embeddings:
                embeddings[concept]= np.zeros((dim,))

            if concept in self.cpnet_ent2id.keys():
                embeddings[concept] = self.cpnet_ent_embs[concept]
                overlap_concept.add(concept)
            else:
                embeddings[concept] = self.swow_ent_embs[concept]

        ent_embs = np.array(list(embeddings.values()), dtype="float32")
        rel_embs = np.array(list(self.cpnet_rel_embs.values()), dtype="float32")

        ent_embeddings_file = config["paths_{}".format(args.kg_name)]["concept_vec_npy_cpnet"]
        rel_embeddings_file = config["paths_{}".format(args.kg_name)]["relation_vec_npy_cpnet"]

        check_path(ent_embeddings_file)

        np.save(ent_embeddings_file, ent_embs)
        np.save(rel_embeddings_file, rel_embs)
        print("Overlap concepts: {}".format(len(overlap_concept)))
        print("Save {}".format(ent_embeddings_file))
        print("Save {}".format(rel_embeddings_file))
        return ent_embs, rel_embs


    def predict_relations(self, ent_embs, rel_embs):
        '''
        input: train2id, or, swow_2relations.en.csv
        '''
        num_count = 0
        triples = []
        triples_seen = set()
        triples_text = []

        new_relations_dict = dict()
        input_path_csv = config["paths_{}".format(args.kg_name)]["triples_input"]
        print(self.cpnet_id2rel)

        ent_embs = ent_embs/np.linalg.norm(ent_embs, ord=2, axis=1, keepdims=True)
        rel_embs = rel_embs/np.linalg.norm(rel_embs, ord=2, axis=1, keepdims=True)
        dist_threshold = 1.40
        with open(input_path_csv, "r", encoding="utf8") as f:
            # first line is the triple num
            lines =  f.readlines()
            for line in tqdm(lines, total= len(lines), desc="predicting relations"):
                num_count +=1
                line = line.strip()
                rel, head, tail, weight  = line.split("\t")

                head_id = self.swow_ent2id[head]
                tail_id = self.swow_ent2id[tail]
                rel_id = self.swow_rel2id[rel]

                head_emb = ent_embs[head_id]
                tail_emb = ent_embs[tail_id]

                dist = np.linalg.norm(head_emb + rel_embs - tail_emb, axis=1) #(34,d)
                k = 1
                dist_sort = np.argsort(dist)[:k]
                pred_rel_id = dist_sort[0]
                pred_rel_dist = dist[pred_rel_id]

                pred_rel = self.cpnet_id2rel[pred_rel_id]

                if num_count <2:
                    print(pred_rel_id, pred_rel, pred_rel_dist)

                if (rel, head, tail) not in triples_seen:
                    triples_seen.add((rel, head, tail))
                    triples_text.append((rel, head, tail, weight))

                for i, k in enumerate(dist_sort):
                    rel_k = self.cpnet_id2rel[k]

                    if pred_rel_dist < dist_threshold and (rel_k, head, tail) not in triples_seen:
                        triples_seen.add((rel_k, head, tail))
                        triples_text.append((rel_k, head, tail, weight ))
                        if rel_k not in new_relations_dict:
                            new_relations_dict[rel_k] = 1
                        else:
                            new_relations_dict[rel_k] += 1


        print("New relations  {}".format(new_relations_dict))
        print("New triples num {}".format(sum(new_relations_dict.values())))

        #check_path(output_path_csv)
        #with open(output_path_csv, "w", encoding="utf8") as fw:
        #    fw.write("%d\n" % len(triples_text))
        #    for t in triples_text:
        #        fw.write("{}\t{}\t{}\t{}\n".format(t[0], t[1], t[2], t[3]))
        #print("Save {} with {} triples".format(output_path_csv, len(triples_text)))
        return triples_text


    def write_files(self, triples,
                    output_csv_path, output_vocab_path, output_relation_path):
        '''
        input: (rel, heat, tail, freq)
        '''
        cpnet_vocab = []
        cpnet_vocab.append(PAD_TOKEN)

        concepts_seen = set()
        relation_vocab = set()
        check_path(output_csv_path)
        fout = open(output_csv_path, "w", encoding="utf8")
        triples = list(triples)
        cnt=0
        for (rel, head, tail, freq) in triples:
            fout.write('\t'.join([rel, head, tail, str(freq)]) + '\n')
            cnt+=1
            relation_vocab.add(rel)
            for w in [head, tail]:
                if w not in concepts_seen:
                    concepts_seen.add(w)
                    cpnet_vocab.append(w)

        check_path(output_vocab_path)
        with open(output_vocab_path, 'w') as fout:
            for word in cpnet_vocab:
                fout.write(word + '\n')

        relation_list = list(relation_vocab)
        with open(output_relation_path, 'w') as fout:
            for word in relation_list:
                fout.write(word + '\n')

        print(f'extracted %d triples to %s'%(cnt, output_csv_path))
        print(f'extracted %d concpet vocabulary to %s'%(len(cpnet_vocab), output_vocab_path))
        print(f'extracted %d relatinos to %s'%(len(relation_list), output_relation_path))
        print()

######## Utils Functions #######
def load_embs_from_npy(vec_path, vocab_path):
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = [l.strip() for l in f.readlines()]
        vocab2id = {word:i for i, word in enumerate(vocab)}

    vectors = np.load(vec_path)
    assert(len(vectors) == len(vocab))

    embeddings = {}
    for i in range(0, len(vectors)):
        embeddings[vocab[i]] = vectors[i]

    assert(vocab2id.keys() == embeddings.keys())
    #print(set(vocab2id.keys()) - set(embeddings.keys()))

    print("Read " + str(len(embeddings)) + " vectors.")
    return vocab2id, embeddings

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)



if __name__=='__main__':
  #ent_embs, rel_embs = create_pretrained_concept_embs()
  #predict_relations(ent_embs, rel_embs)
  PredictRelations()



