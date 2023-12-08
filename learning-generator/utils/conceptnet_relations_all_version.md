
https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz

# CN 5.5.0: 
extracting English concepts and relations from ConceptNet...100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 28860811/28860811 [01:25<00:00, 338980.05it/s]
relation types: 46
dict_keys(['antonym', 'atlocation', 'capableof', 'causes', 'causesdesire', 'createdby', 'definedas', 'derivedfrom', 'desires', 'distinctfrom', 'entails', 'etymologicallyrelatedto', 'formof', 'hasa', 'hascontext', 'hasfirstsubevent', 'haslastsubevent', 'hasprerequisite', 'hasproperty', 'hassubevent', 'instanceof', 'isa', 'locatednear', 'madeof', 'mannerof', 'motivatedbygoal', 'notcapableof', 'notdesires', 'nothasproperty', 'partof', 'receivesaction', 'relatedto', 'similarto', 'symbolof', 'synonym', 'usedfor', 'capital', 'field', 'genre', 'genus', 'influencedby', 'knownfor', 'language', 'leader', 'occupation', 'product'])


# CN 5.6.0 
(adding: etymologicallyderivedfrom)
extracting English concepts and relations from ConceptNet...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32755210/32755210 [01:55<00:00, 284097.71it/s]
relation types: 47
dict_keys(['antonym', 'atlocation', 'capableof', 'causes', 'causesdesire', 'createdby', 'definedas', 'derivedfrom', 'desires', 'distinctfrom', 'entails', 'etymologicallyderivedfrom', 'etymologicallyrelatedto', 'formof', 'hasa', 'hascontext', 'hasfirstsubevent', 'haslastsubevent', 'hasprerequisite', 'hasproperty', 'hassubevent', 'instanceof', 'isa', 'locatednear', 'madeof', 'mannerof', 'motivatedbygoal', 'notcapableof', 'notdesires', 'nothasproperty', 'partof', 'receivesaction', 'relatedto', 'similarto', 'symbolof', 'synonym', 'usedfor', 'capital', 'field', 'genre', 'genus', 'influencedby', 'knownfor', 'language', 'leader', 'occupation', 'product'])

discard_relations=('relatedto' 'synonym', 'antonym', 'derivedfrom', 'formof', 'etymologicallyderivedfrom','etymologicallyrelatedto', 'language','capital', 'field', 'genre', 'genus', 'knownfor', 'leader', 'occupation', 'product', 'notdesires', 'nothasproperty','notcapableof')

relation_groups=[
    'isa/hasproperty/madeof/partof/definedas/instanceof/hasa/createdby',
    'atlocation/locatednear/hascontext/similarto/symbolof'
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'causes/causesdesire/motivatedbygoal/desires/influencedby',
    'usedfor/receivesaction',
    'capableof',
    'distinctfrom',
]
