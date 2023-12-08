
import argparse

def add_basic_arguments(parser):
    parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file', type=str, default='./data/swow/swow_triple_freq2.filter')
    parser.add_argument('--input_order', type=str, default="rht")
    parser.add_argument('--match_mode', type=str, default="hard", choices=["hard", "total_soft"])
    parser.add_argument('--align_dir',type=str, default="data/alignment/C_S_V0.1")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--pivot_kg', type=str, default='ConceptNet')

def add_ground_arguments(parser):
    parser.add_argument('--write_ground_triples', action='store_true')
    parser.add_argument('--output_csv_path_cn', type=str)
    parser.add_argument('--output_vocab_path_cn', type=str)
    parser.add_argument('--output_relation_path_cn', type=str)

    parser.add_argument('--output_csv_path_sw', type=str)

    parser.add_argument('--output_csv_path_non_overlap', type=str)
    parser.add_argument('--output_vocab_path_non_overlap', type=str)
    parser.add_argument('--output_relation_path_non_overlap', type=str)

    parser.add_argument('--add_isphrase_rel', action='store_true')
    parser.add_argument('--write_non_overlap', action='store_true')

    parser.add_argument('--swap_retrieval', action='store_true', help='swap head and tail to retrieve edge when finding overlap edges')


def add_alignment_arguments(parser):
    parser.add_argument('--out_dir',type=str, default="data/alignment/C_S_V0.1")

    parser.add_argument('--output_csv_path', type=str)
    parser.add_argument('--output_vocab_path', type=str)
    parser.add_argument('--output_relation_path', type=str)

    parser.add_argument('--add_cn_triples_to_swow', action='store_true')
    parser.add_argument('--sample_node_size', type=int, default=3000, help='node number for test set')
    parser.add_argument('--swow_prefix', type=str, default='swow_3rel_freq1', help='swow kg types')
    

def get_parser():
    """A helper function that handles the arguments that all models share"""
    parser = argparse.ArgumentParser(add_help=False)
    add_basic_arguments(parser)
    add_ground_arguments(parser)
    add_alignment_arguments(parser)
    # add_additional_arguments(parser)
    
    return parser
