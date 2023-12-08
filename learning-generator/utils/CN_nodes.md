CN5.5.0
graph file saved to data/conceptnet_debug/conceptnet_graph.nx
num of nodes: 305754
num of edges: 997128

save data/conceptnet_debug/relation_vocab.pkl
save data/conceptnet_debug/entity_vocab.pkl
Finish
num of nodes: 382220
num of edges: 1212454


reported:
num of nodes: 382220
num of edges: 1212454

prune True, CN5.6
num of nodes: 288301
num of edges: 949900

prune False, CN5.6
num of nodes: 406802
num of edges: 1364954




7 relations
num of entities: 390570
num of relations: 14
loading kg
loading knoweldge graph....
num of nodes: 390562
num of edges: 1283570


discard_relations=('relatedto','synonym', 'antonym', 'derivedfrom', 'formof', 'etymologicallyderivedfrom','etymologicallyrelatedto', 'language','capital', 'field', 'genre', 'genus', 'knownfor', 'leader', 'occupation', 'product', 'notdesires', 'nothasproperty','notcapableof')

relation_groups=[
    'isa/hasproperty/madeof/partof/definedas/instanceof/hasa/createdby',
    'atlocation/locatednear/hascontext/similarto/symbolof',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'causes/causesdesire/motivatedbygoal/desires/influencedby',
    'usedfor/receivesaction',
    'capableof',
    'distinctfrom',
]

merged_relations = [
    'isa',
    'atlocation',
    'hassubevent',
    'causes',
    'usedfor',
    'capableof',
    'distinctfrom',
]