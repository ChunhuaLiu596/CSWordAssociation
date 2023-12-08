from conceptnet_4_path import load_vocab

relations_reported={'atlocation': 0, 'capableof': 1, 'causes': 2, 'causesdesire': 3, 'createdby': 4, 'definedas': 5, 'desires': 6, 'distinctfrom': 7, 'entails': 8, 'hasa': 9, 'hascontext': 10, 'hasfirstsubevent': 11, 'haslastsubevent': 12, 'hasprerequisite': 13, 'hasproperty': 14, 'hassubevent': 15, 'instanceof': 16, 'isa': 17, 'locatednear': 18, 'madeof': 19, 'mannerof': 20, 'motivatedbygoal': 21, 'notcapableof': 22, 'notdesires': 23, 'nothasproperty': 24, 'partof': 25, 'receivesaction': 26, 'similarto': 27, 'symbolof': 28, 'usedfor': 29, 'capital': 30, 'field': 31, 'genre': 32, 'genus': 33, 'influencedby': 34, 'knownfor': 35, 'language': 36, 'leader': 37, 'occupation': 38, 'product': 39, '_atlocation': 40, '_capableof': 41, '_causes': 42, '_causesdesire': 43, '_createdby': 44, '_definedas': 45, '_desires': 46, '_distinctfrom': 47, '_entails': 48, '_hasa': 49, '_hascontext': 50, '_hasfirstsubevent': 51, '_haslastsubevent': 52, '_hasprerequisite': 53, '_hasproperty': 54, '_hassubevent': 55, '_instanceof': 56, '_isa': 57, '_locatednear': 58, '_madeof': 59, '_mannerof': 60, '_motivatedbygoal': 61, '_notcapableof': 62, '_notdesires': 63, '_nothasproperty': 64, '_partof': 65, '_receivesaction': 66, '_similarto': 67, '_symbolof': 68, '_usedfor': 69, '_capital': 70, '_field': 71, '_genre': 72, '_genus': 73, '_influencedby': 74, '_knownfor': 75, '_language': 76, '_leader': 77, '_occupation': 78, '_product': 79}

discared_relations_reported= ('relatedto', ' synonym', ' antonym', ' derivedfrom', ' formof', ' etymologicallyderivedfrom', 'etymologicallyrelatedto')

relation_groups_base=[
    'isa/hasproperty/madeof/partof/definedas/instanceof/hasa/createdby',
    'atlocation/locatednear/hascontext/similarto/symbolof',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'causes/causesdesire/motivatedbygoal/desires/influencedby',
    'usedfor/receivesaction',
    'capableof',
    'distinctfrom',
]
discard_relations=('relatedto', 'synonym', 'antonym', 'derivedfrom', 'formof', 'etymologicallyderivedfrom','etymologicallyrelatedto','capital', 'field', 'genre', 'genus', 'knownfor', 'language', 'leader', 'occupation', 'product', 'notdesires', 'nothasproperty','notcapableof')


def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups_base:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping

relation_mapping = load_merge_relation()
print("intersection:", set(relation_mapping).intersection(set(discard_relations)))
print("selected relations {}".format(len(relation_mapping.keys())))
print("discard relations {}".format(len(discard_relations)))

data_dir='data/conceptnet/'
i2r, r2i, i2e, e2i = load_vocab(data_dir)

rel_cn56= set(['antonym', 'atlocation', 'capableof', 'causes', 'causesdesire', 'createdby', 'definedas', 'derivedfrom', 'desires', 'distinctfrom', 'entails', 'etymologicallyderivedfrom', 'etymologicallyrelatedto', 'formof', 'hasa', 'hascontext', 'hasfirstsubevent', 'haslastsubevent', 'hasprerequisite', 'hasproperty', 'hassubevent', 'instanceof', 'isa', 'locatednear', 'madeof', 'mannerof', 'motivatedbygoal', 'notcapableof', 'notdesires', 'nothasproperty', 'partof', 'receivesaction', 'relatedto', 'similarto', 'symbolof', 'synonym', 'usedfor', 'capital', 'field', 'genre', 'genus', 'influencedby', 'knownfor', 'language', 'leader', 'occupation', 'product'])
rel_local = set(discard_relations)| set(relation_mapping)
print("intersection", set(discard_relations).intersection(set(relation_mapping)))
print("difference:", set(rel_local).difference(rel_cn56))
print("difference:", set(rel_cn56).difference(rel_local))

# intersection: {'symbolof', 'hassubevent', 'relatedto', 'synonym'}
# print(i2r[:40])

# print( "difference", set(relation_mapping.keys()) | set(discard_relations) - set(i2r[:40]))
# print(r2i)




