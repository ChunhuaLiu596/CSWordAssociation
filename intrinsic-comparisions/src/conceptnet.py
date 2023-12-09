import argparse
from utils import is_stopword
import json
import tqdm

class ConceptNet:

    def __init__(self, path, delimiter="\t"):
        self.triples = {}
        self.nodes=set()
        self.num_triples = 0
        self.relation_count={}
        for triple in open(path, 'r', encoding='utf-8'):
            split_data = triple.strip().split(delimiter)
            if len(split_data)==3:
                arg1, r, arg2 =  split_data
            elif len(split_data)==4:
                arg1, r, arg2, frequency = split_data

            if not arg1 in self.triples:
                self.triples[arg1] = {}
            self.triples[arg1][arg2] = r

            if not r in self.relation_count:
                self.relation_count[r]= 0
            self.relation_count[r] +=1

            self.num_triples += 1
            self.nodes.add(arg1)
            self.nodes.add(arg2)
        num_triples_check = sum(len(v) for v in self.triples.values())
        assert self.num_triples==num_triples_check, "num_triples: {}, num_triples_check: {}".format(self.num_triples, num_triples_check) 
        self.relation_count = sorted(self.relation_count.items(), key=lambda x: x[1])
        print('Loaded %d triples from %s with %d vocab' % (self.num_triples, path, len(self.nodes)))
        print('Relation count: total {} relations. {}'.format(len(self.relation_count), self.relation_count))

    def get_relation(self, w1, w2):
        if not w1 in self.triples:
            return '<NULL>'
        return self.triples[w1].get(w2, '<NULL>')

    def get_relation_fuzzy(self, w1, w2, fuzzy_match=False):
        if not w1 in self.triples:
            return '<NULL>'
        rel = self.triples[w1].get(w2,'<NULL>')

        if fuzzy_match and rel=='<NULL>':
            w2_phrase = self.triples[w1].keys()
            for k in w2_phrase:
                if w2 in k.split():
                    rel = self.triples[w1][k]
                    break
        return rel

    def p_q_relation(self, passage, query):
        passage = [w.lower() for w in passage]
        query = [w.lower() for w in query]
        query = set(query) | set([' '.join(query[i:(i+2)]) for i in range(len(query))])
        query = set([q for q in query if not is_stopword(q)])
        ret = ['<NULL>' for _ in passage]
        for i in range(len(passage)):
            for q in query:
                r = self.get_relation(passage[i], q)
                if r != '<NULL>':
                    ret[i] = r
                    break
                r = self.get_relation(' '.join(passage[i:(i+2)]), q)
                if r != '<NULL>':
                    ret[i] = r
                    break
        return ret


def preprocess_conceptnet(path,out_path,weight_threshold=0.0, delimiter='_'):
    #import utils
    writer = open(out_path, 'w', encoding='utf-8')
    weight_values=set()

    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]

    for line in open(path, 'r', encoding='utf-8'):
        fs = line.split('\t')
        relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]

        lan1, w1 = _get_lan_and_w(arg1)
        if lan1 != 'en':
            continue

        lan2, w2 = _get_lan_and_w(arg2)
        if lan2 != 'en':
            continue

        obj = json.loads(fs[-1])
        if obj['weight'] < weight_threshold:
            weight_values.add(obj['weight'])
            continue

        w1 = ' '.join(w1.lower().split('_'))
        w2 = ' '.join(w2.lower().split('_'))
        #writer.write('%s\t%s\t%s\t%s\n' % (relation, w1, w2,obj['weight']))
        writer.write('%s\t%s\t%s\n' % (relation, w1, w2))

    #print("weight_values:%s"%(weight_values))
    print("Writed file %s"%(out_path))
    writer.close()

def convert_format(input_file, output_file, delimiter="\t"):
    '''
    input_format: r, h, t, weight or r,h,t
    output_format: h, r, t
    '''
    writer = open(output_file, 'w', encoding='utf-8')
    for triple in tqdm.tqdm(open(input_file, 'r', encoding='utf-8')):
        split_data = triple.strip().split(delimiter)
        if len(split_data)==3:
            r, arg1, arg2 =  split_data
        elif len(split_data)==4:
            r, arg1, arg2, frequency = split_data
        out_line="{}\t{}\t{}\n".format(arg1, r, arg2)
        writer.write(out_line)



#concept_net = ConceptNet()
if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--process_conceptnet',action='store_true')
    parser.add_argument('--conceptnet_raw_file',type=str, default='../../0.Dataset/conceptnet-assertions-5.7.0.csv')
    parser.add_argument('--conceptnet_quadruple',type=str, default='./data/concept.filter')
    parser.add_argument('--conceptnet_weight_threshold',type=float, default=0.0)

    parser.add_argument('--generate_rel_triples',action='store_true')
    parser.add_argument('--concepenet_rel_triples',type=str, default='./data/conceptnet/conceptnet_rel_triples')
    args = parser.parse_args()
    #print(args)
    if args.process_conceptnet:
        print("preprocessing the conceptnet raw file ...")
        preprocess_conceptnet(args.conceptnet_raw_file,args.conceptnet_quadruple,args.conceptnet_weight_threshold)

    if args.generate_rel_triples:
        convert_format(args.concepenet_quadruple, args.concepenet_rel_triples)
