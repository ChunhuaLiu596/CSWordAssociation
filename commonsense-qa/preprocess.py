import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_obqa import convert_to_obqa_statement
from utils.convert_mcscript import convert_to_mcscript_statement
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from utils.embedding import glove2npy, load_pretrained_embeddings
from utils.grounding import create_matcher_patterns, ground
from utils.paths import find_paths, score_paths, prune_paths, find_relational_paths_from_paths, generate_path_and_graph_from_adj
from utils.graph import generate_graph, generate_adj_data_from_grounded_concepts, coo_to_normalized
from utils.triples import generate_triples_from_adj

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('kg_name', type=str)
parser.add_argument('--run', default=['common', 'csqa'], choices=['common', 'csqa', 'obqa', 'mcscript','make_word_vocab'], nargs='+')
parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--debug', action='store_true', help='enable debug mode')

args = parser.parse_args()
kg_name=args.kg_name

if kg_name in ('cpnet', 'cpnet7rel', 'cpnet1rel'):
    from utils.conceptnet import extract_english, construct_graph
elif kg_name in ('swow', 'swow1rel'):
    from utils.swow import extract_english, construct_graph
elif kg_name in ('cpnet_swow', 'cpnet_swow_1rel'):
    from utils.conceptnet_swow import extract_english, construct_graph

input_paths = {
   'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
    'mcscript': {
        'train': './data/mcscript/train-data.json',
        'dev': './data/mcscript/dev-data.json',
        'test': './data/mcscript/test-data.json',
    },
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
        # 'csv': './data/cpnet/conceptnet-assertions-debug.csv',
    },
    'cpnet7rel': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'cpnet1rel': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'swow': {
        'csv': './data/swow/SWOW-EN.R100.csv',
    },
    'swow1rel': {
        'csv': './data/swow/SWOW-EN.R100.csv',
    },
    'glove': {
        'txt': './data/glove/glove.6B.300d.txt',
    },
    'numberbatch': {
        'txt': './data/transe/numberbatch-en-19.08.txt',
    },
    'transe': {
        'ent': './data/transe/glove.transe.sgd.ent.npy',
        'rel': './data/transe/glove.transe.sgd.rel.npy',
    },
}

output_paths = {
    f'{kg_name}': {
        'csv': f'./data/{kg_name}/conceptnet.en.csv',
        'vocab': f'./data/{kg_name}/concept.txt',
        'patterns': f'./data/{kg_name}/matcher_patterns.json',
        'unpruned-graph': f'./data/{kg_name}/conceptnet.en.unpruned.graph',
        'pruned-graph': f'./data/{kg_name}/conceptnet.en.pruned.graph',
    },
    'glove': {
        'npy': './data/glove/glove.6B.300d.npy',
        'vocab': './data/glove/glove.vocab',
    },
    'numberbatch': {
        'npy': './data/transe/nb.npy',
        'vocab': './data/transe/nb.vocab',
        'concept_npy': './data/transe/concept.nb.npy'
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
            'vocab': './data/csqa/statement/vocab.json',
        },
        'statement-with-ans-pos': {
            'train': './data/csqa/statement/train.statement-with-ans-pos.jsonl',
            'dev': './data/csqa/statement/dev.statement-with-ans-pos.jsonl',
            'test': './data/csqa/statement/test.statement-with-ans-pos.jsonl',
        },
        'tokenized': {
            'train': './data/csqa/tokenized/train.tokenized.txt',
            'dev': './data/csqa/tokenized/dev.tokenized.txt',
            'test': './data/csqa/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': f'./data/csqa/{kg_name}/grounded/train.grounded.jsonl',
            # 'dev': f'./data/csqa/{kg_name}/grounded/dev.grounded.debug.jsonl',
            'dev': f'./data/csqa/{kg_name}/grounded/dev.grounded.jsonl',
            'test': f'./data/csqa/{kg_name}/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': f'./data/csqa/{kg_name}/paths/train.paths.raw.jsonl',
            'raw-dev': f'./data/csqa/{kg_name}/paths/dev.paths.raw.jsonl',
            'raw-test': f'./data/csqa/{kg_name}/paths/test.paths.raw.jsonl',
            'scores-train': f'./data/csqa/{kg_name}/paths/train.paths.scores.jsonl',
            'scores-dev': f'./data/csqa/{kg_name}/paths/dev.paths.scores.jsonl',
            'scores-test': f'./data/csqa/{kg_name}/paths/test.paths.scores.jsonl',
            'pruned-train': f'./data/csqa/{kg_name}/paths/train.paths.pruned.jsonl',
            'pruned-dev': f'./data/csqa/{kg_name}/paths/dev.paths.pruned.jsonl',
            'pruned-test': f'./data/csqa/{kg_name}/paths/test.paths.pruned.jsonl',
            'adj-train': f'./data/csqa/{kg_name}/paths/train.paths.adj.jsonl',
            'adj-dev': f'./data/csqa/{kg_name}/paths/dev.paths.adj.jsonl',
            'adj-test': f'./data/csqa/{kg_name}/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': f'./data/csqa/{kg_name}/graph/train.graph.jsonl',
            'dev': f'./data/csqa/{kg_name}/graph/dev.graph.jsonl',
            'test': f'./data/csqa/{kg_name}/graph/test.graph.jsonl',
            'adj-train': f'./data/csqa/{kg_name}/graph/train.graph.adj.pk',
            'adj-dev': f'./data/csqa/{kg_name}/graph/dev.graph.adj.pk',
            'adj-test': f'./data/csqa/{kg_name}/graph/test.graph.adj.pk',
            'nxg-from-adj-train': f'./data/csqa/{kg_name}/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': f'./data/csqa/{kg_name}/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': f'./data/csqa/{kg_name}/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': f'./data/csqa/{kg_name}/triples/train.triples.pk',
            'dev': f'./data/csqa/{kg_name}/triples/dev.triples.pk',
            'test': f'./data/csqa/{kg_name}/triples/test.triples.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
            'train-fairseq': './data/obqa/fairseq/official/train.jsonl',
            'dev-fairseq': './data/obqa/fairseq/official/valid.jsonl',
            'test-fairseq': './data/obqa/fairseq/official/test.jsonl',
            'vocab': './data/obqa/statement/vocab.json',
        },
        'tokenized': {
            'train': './data/obqa/tokenized/train.tokenized.txt',
            'dev': './data/obqa/tokenized/dev.tokenized.txt',
            'test': './data/obqa/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': f'./data/obqa/{kg_name}/grounded/train.grounded.jsonl',
            'dev': f'./data/obqa/{kg_name}/grounded/dev.grounded.jsonl',
            'test': f'./data/obqa/{kg_name}/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': f'./data/obqa/{kg_name}/paths/train.paths.raw.jsonl',
            'raw-dev': f'./data/obqa/{kg_name}/paths/dev.paths.raw.jsonl',
            'raw-test': f'./data/obqa/{kg_name}/paths/test.paths.raw.jsonl',
            'scores-train': f'./data/obqa/{kg_name}/paths/train.paths.scores.jsonl',
            'scores-dev': f'./data/obqa/{kg_name}/paths/dev.paths.scores.jsonl',
            'scores-test': f'./data/obqa/{kg_name}/paths/test.paths.scores.jsonl',
            'pruned-train': f'./data/obqa/{kg_name}/paths/train.paths.pruned.jsonl',
            'pruned-dev': f'./data/obqa/{kg_name}/paths/dev.paths.pruned.jsonl',
            'pruned-test': f'./data/obqa/{kg_name}/paths/test.paths.pruned.jsonl',
            'adj-train': f'./data/obqa/{kg_name}/paths/train.paths.adj.jsonl',
            'adj-dev': f'./data/obqa/{kg_name}/paths/dev.paths.adj.jsonl',
            'adj-test': f'./data/obqa/{kg_name}/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': f'./data/obqa/{kg_name}/graph/train.graph.jsonl',
            'dev': f'./data/obqa/{kg_name}/graph/dev.graph.jsonl',
            'test': f'./data/obqa/{kg_name}/graph/test.graph.jsonl',
            'adj-train': f'./data/obqa/{kg_name}/graph/train.graph.adj.pk',
            'adj-dev': f'./data/obqa/{kg_name}/graph/dev.graph.adj.pk',
            'adj-test': f'./data/obqa/{kg_name}/graph/test.graph.adj.pk',
            'nxg-from-adj-train': f'./data/obqa/{kg_name}/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': f'./data/obqa/{kg_name}/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': f'./data/obqa/{kg_name}/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': f'./data/obqa/{kg_name}/triples/train.triples.pk',
            'dev': f'./data/obqa/{kg_name}/triples/dev.triples.pk',
            'test': f'./data/obqa/{kg_name}/triples/test.triples.pk',
        },
    },
    'mcscript': {
        'statement': {
            'train': './data/mcscript/statement/train.statement.jsonl',
            'dev': './data/mcscript/statement/dev.statement.jsonl',
            'test': './data/mcscript/statement/test.statement.jsonl',
            'vocab': './data/mcscript/statement/vocab.json',
        },
        'tokenized': {
            'train': './data/mcscript/tokenized/train.tokenized.txt',
            'dev': './data/mcscript/tokenized/dev.tokenized.txt',
            'test': './data/mcscript/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': f'./data/mcscript/{kg_name}/grounded/train.grounded.jsonl',
            'dev': f'./data/mcscript/{kg_name}/grounded/dev.grounded.jsonl',
            'test': f'./data/mcscript/{kg_name}/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': f'./data/mcscript/{kg_name}/paths/train.paths.raw.jsonl',
            'raw-dev': f'./data/mcscript/{kg_name}/paths/dev.paths.raw.jsonl',
            'raw-test': f'./data/mcscript/{kg_name}/paths/test.paths.raw.jsonl',
            'scores-train': f'./data/mcscript/{kg_name}/paths/train.paths.scores.jsonl',
            'scores-dev': f'./data/mcscript/{kg_name}/paths/dev.paths.scores.jsonl',
            'scores-test': f'./data/mcscript/{kg_name}/paths/test.paths.scores.jsonl',
            'pruned-train': f'./data/mcscript/{kg_name}/paths/train.paths.pruned.jsonl',
            'pruned-dev': f'./data/mcscript/{kg_name}/paths/dev.paths.pruned.jsonl',
            'pruned-test': f'./data/mcscript/{kg_name}/paths/test.paths.pruned.jsonl',
            'adj-train': f'./data/mcscript/{kg_name}/paths/train.paths.adj.jsonl',
            'adj-dev': f'./data/mcscript/{kg_name}/paths/dev.paths.adj.jsonl',
            'adj-test': f'./data/mcscript/{kg_name}/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': f'./data/mcscript/{kg_name}/graph/train.graph.jsonl',
            'dev': f'./data/mcscript/{kg_name}/graph/dev.graph.jsonl',
            'test': f'./data/mcscript/{kg_name}/graph/test.graph.jsonl',
            'adj-train': f'./data/mcscript/{kg_name}/graph/train.graph.adj.pk',
            'adj-dev': f'./data/mcscript/{kg_name}/graph/dev.graph.adj.pk',
            'adj-test': f'./data/mcscript/{kg_name}/graph/test.graph.adj.pk',
            'nxg-from-adj-train': f'./data/mcscript/{kg_name}/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': f'./data/mcscript/{kg_name}/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': f'./data/mcscript/{kg_name}/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': f'./data/mcscript/{kg_name}/triples/train.triples.pk',
            'dev': f'./data/mcscript/{kg_name}/triples/dev.triples.pk',
            'test': f'./data/mcscript/{kg_name}/triples/test.triples.pk',
        },
    },
}


def main():
   
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            # {'func': glove2npy, 'args': (input_paths['glove']['txt'], output_paths['glove']['npy'], output_paths['glove']['vocab'])},
            # {'func': glove2npy, 'args': (input_paths['numberbatch']['txt'], output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], True)},
            # {'func': load_pretrained_embeddings,
            #  'args': (output_paths['numberbatch']['npy'], output_paths['numberbatch']['vocab'], output_paths[kg_name]['vocab'], False, output_paths['numberbatch']['concept_npy'])},
            # {'func': extract_english, 'args': (input_paths[kg_name]['csv'], output_paths[kg_name]['csv'], output_paths[kg_name]['vocab'], kg_name)},
            #{'func': construct_graph, 'args': (output_paths[kg_name]['csv'], output_paths[kg_name]['vocab'],
            #                                   output_paths[kg_name]['unpruned-graph'], False, kg_name)},
            #{'func': construct_graph, 'args': (output_paths[kg_name]['csv'], output_paths[kg_name]['vocab'],
            #                                   output_paths[kg_name]['pruned-graph'], True, kg_name)},
            {'func': create_matcher_patterns, 'args': (output_paths[kg_name]['vocab'], output_paths[kg_name]['patterns'])},
        ],
        'csqa': [
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            # {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},

            # {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths[kg_name]['vocab'],
                                    #   output_paths[kg_name]['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths[kg_name]['vocab'],
                                    #   output_paths[kg_name]['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            # {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths[kg_name]['vocab'],
                                    #   output_paths[kg_name]['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
# 
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['dev'], output_paths[kg_name]['pruned-graph'],
                                                                    #    output_paths[kg_name]['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['test'], output_paths[kg_name]['pruned-graph'],
                                                                    #    output_paths[kg_name]['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['csqa']['grounded']['train'], output_paths[kg_name]['pruned-graph'],
                                                                        output_paths[kg_name]['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},

            # {'func': find_paths, 'args': (output_paths['csqa']['grounded']['dev'], output_paths[kg_name]['vocab'],
            #                                output_paths[kg_name]['unpruned-graph'], output_paths['csqa']['paths']['raw-dev'], args.nprocs, args.seed)},

            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['csqa']['graph']['adj-train'], output_paths[kg_name]['pruned-graph'], output_paths['csqa']['paths']['adj-train'], output_paths['csqa']['graph']['nxg-from-adj-train'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['csqa']['graph']['adj-dev'], output_paths[kg_name]['pruned-graph'], output_paths['csqa']['paths']['adj-dev'], output_paths['csqa']['graph']['nxg-from-adj-dev'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['csqa']['graph']['adj-test'], output_paths[kg_name]['pruned-graph'], output_paths['csqa']['paths']['adj-test'], output_paths['csqa']['graph']['nxg-from-adj-test'], args.nprocs)},
            # {'func': generate_triples_from_adj, 'args': (output_paths['csqa']['graph']['adj-train'], output_paths['csqa']['grounded']['train'],
                                                        #  output_paths[kg_name]['vocab'], output_paths['csqa']['triple']['train'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['csqa']['graph']['adj-dev'], output_paths['csqa']['grounded']['dev'],
                                                        #  output_paths[kg_name]['vocab'], output_paths['csqa']['triple']['dev'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['csqa']['graph']['adj-test'], output_paths['csqa']['grounded']['test'],
                                                        #  output_paths[kg_name]['vocab'], output_paths['csqa']['triple']['test'])},
        ],

        'obqa': [
            # {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'], output_paths['obqa']['statement']['train-fairseq'])},
            # {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'], output_paths['obqa']['statement']['dev-fairseq'])},
            # {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'], output_paths['obqa']['statement']['test-fairseq'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths[kg_name]['vocab'],
                                      output_paths[kg_name]['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths[kg_name]['vocab'],
                                      output_paths[kg_name]['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths[kg_name]['vocab'],
                                      output_paths[kg_name]['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['obqa']['grounded']['train'], output_paths[kg_name]['pruned-graph'],
            #                                                             output_paths[kg_name]['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['obqa']['grounded']['dev'], output_paths[kg_name]['pruned-graph'],
            #                                                             output_paths[kg_name]['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            # {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['obqa']['grounded']['test'], output_paths[kg_name]['pruned-graph'],
            #                                                             output_paths[kg_name]['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
            # {'func': generate_triples_from_adj, 'args': (output_paths['obqa']['graph']['adj-train'], output_paths['obqa']['grounded']['train'],
            #                                              output_paths[kg_name]['vocab'], output_paths['obqa']['triple']['train'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['obqa']['graph']['adj-dev'], output_paths['obqa']['grounded']['dev'],
            #                                              output_paths[kg_name]['vocab'], output_paths['obqa']['triple']['dev'])},
            # {'func': generate_triples_from_adj, 'args': (output_paths['obqa']['graph']['adj-test'], output_paths['obqa']['grounded']['test'],
            #                                              output_paths[kg_name]['vocab'], output_paths['obqa']['triple']['test'])},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['obqa']['graph']['adj-train'], output_paths[kg_name]['pruned-graph'], output_paths['obqa']['paths']['adj-train'], output_paths['obqa']['graph']['nxg-from-adj-train'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['obqa']['graph']['adj-dev'], output_paths[kg_name]['pruned-graph'], output_paths['obqa']['paths']['adj-dev'], output_paths['obqa']['graph']['nxg-from-adj-dev'], args.nprocs)},
            # {'func': generate_path_and_graph_from_adj, 'args': (output_paths['obqa']['graph']['adj-test'], output_paths[kg_name]['pruned-graph'], output_paths['obqa']['paths']['adj-test'], output_paths['obqa']['graph']['nxg-from-adj-test'], args.nprocs)},
        ],
        'mcscript': [
            {'func': convert_to_mcscript_statement, 'args': (input_paths['mcscript']['train'], output_paths['mcscript']['statement']['train'])},
            {'func': convert_to_mcscript_statement, 'args': (input_paths['mcscript']['dev'], output_paths['mcscript']['statement']['dev'])},
            {'func': convert_to_mcscript_statement, 'args': (input_paths['mcscript']['test'], output_paths['mcscript']['statement']['test'])},
            {'func': ground, 'args': (output_paths['mcscript']['statement']['dev'], output_paths[kg_name]['vocab'],
                                      output_paths[kg_name]['patterns'], output_paths['mcscript']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['mcscript']['statement']['train'], output_paths[kg_name]['vocab'],
                                      output_paths[kg_name]['patterns'], output_paths['mcscript']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['mcscript']['statement']['test'], output_paths[kg_name]['vocab'],
                                      output_paths[kg_name]['patterns'], output_paths['mcscript']['grounded']['test'], args.nprocs)},
        ],
        'exp': [
            {'func': convert_to_entailment,
             'args': (input_paths['csqa']['train'], output_paths['csqa']['statement-with-ans-pos']['train'], True)},
            {'func': convert_to_entailment,
             'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement-with-ans-pos']['dev'], True)},
            {'func': convert_to_entailment,
             'args': (input_paths['csqa']['test'], output_paths['csqa']['statement-with-ans-pos']['test'], True)},
        ],

        'make_word_vocab': [
            {'func': make_word_vocab, 'args': ((output_paths['csqa']['statement']['train'],), output_paths['csqa']['statement']['vocab'])},
            {'func': make_word_vocab, 'args': ((output_paths['obqa']['statement']['train'],), output_paths['obqa']['statement']['vocab'])},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
