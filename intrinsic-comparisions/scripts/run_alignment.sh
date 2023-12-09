#!/usr/bin/env bash

#<<COMMENT
#------------------

#swow_source_file='../data/swow/SWOW-EN.R100.csv'

# debug='false'
# debug='true'
debug=$1
echo "#Creating alignment set"

while getopts "d:p:" opt;
do
    case ${opt} in
        d) data=$OPTARG ;;
        *) echo "parameter error";;
    esac
done

if [[ ${debug} == "true" ]]; then
    conceptnet_source_file='./data/cpnet/conceptnet.en.csv.debug'
    swow_filter_file='./data/swow/3rel_freq1/swow_3rel_freq1.en.csv.debug'
    output_folder=./data/alignment/C_S
else
    conceptnet_source_file='./data/cpnet/conceptnet.en.csv'
    swow_filter_file='./data/swow/conceptnet.en.csv'
    # conceptnet_overlap_file='./data/analysis/overlap_cn.en.csv'
    union_file='./data/analysis/cpnet47rel_swow2rel/conceptnet.en.csv'
    output_folder=./data/analysis/cpsw_inter/
fi

if [ ! -d "$output_folder" ]; then
    echo "Creating $output_folder"
    mkdir -p "$output_folder"
fi

out_dir="$output_folder"

##1. preprocess the swow file, filter by frequency
# echo generate $swow_filter_file
# python3 src/swow.py\
#     --swow_file $swow_source_file\
#     --wordpairs_frequency 2\
#     --swow_freq_triple_file $swow_filter_file



# : << COMMENT
# 2. build net_cpn and net_swow, get_shared_nodes, shared_edges
# sample_node_size=10000
# echo align [$conceptnet_source_file] and [$swow_filter_file]
# python3.6 utils/kgsrc/align_cpt_sw.py \
#     --conceptnet_source_file $conceptnet_source_file\
#     --swow_source_file $swow_filter_file\
#     --out_dir $out_dir\
#     --input_order rhtw\
#     --match_mode total_soft\
#     --sample_node_size $sample_node_size >debug

# COMMENT
# 

####### 
##3. Only get the shared nodes and edges 
# : <<COMMENT
echo ground [$conceptnet_source_file] and [$swow_filter_file]
python3.6 utils/kgsrc/ground.py \
    --conceptnet_source_file $conceptnet_source_file\
    --swow_source_file $swow_filter_file\
    --input_order rhtw\
    --match_mode total_soft\
    --align_dir $out_dir\
    --write_ground_triples\
    --output_csv_path_cn "$out_dir/conceptnet.en.csv"\
    --output_csv_path_sw "$out_dir/overlap_sw.en.csv"\
    --swap_retrieval\
    # --add_isphrase_rel
# COMMENT

# echo ground [$conceptnet_source_file] and [$swow_filter_file]
# python3.6 utils/kgsrc/ground.py \
#     --conceptnet_source_file $conceptnet_source_file\
#     --swow_source_file $swow_filter_file\
#     --input_order rhtw\
#     --match_mode total_soft\
#     --align_dir $out_dir\
#     --write_ground_triples\
#     --output_csv_path_cn "$out_dir/overlap_cn.en.csv"\
#     --output_csv_path_sw "$out_dir/overlap_sw.en.csv"\
#     --swap_retrieval\
#     --add_isphrase_rel\
#     --pivot_kg "ConceptNet"


#2. get graph statistics 
# echo "#Computing statistics"
# python3.6 utils/kgsrc/graph_statistics.py --dataset "swow" --source_file $swow_filter_file --input_order rhtw 

# python3.6 utils/kgsrc/graph_statistics.py --dataset "conceptnet" --source_file $conceptnet_source_file --input_order rhtw 

# python3.6 utils/kgsrc/graph_statistics.py --dataset "conceptnet" --source_file $conceptnet_overlap_file --input_order rhtw

# python3.6 utils/kgsrc/graph_statistics.py --dataset "conceptnet" --source_file $conceptnet_source_file --input_order rhtw

# python3.6 utils/kgsrc/graph_statistics.py --dataset "conceptnet" --source_file $union_file  --input_order rhtw







