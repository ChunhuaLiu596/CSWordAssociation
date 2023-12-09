
#######
N=2370
output='dev_data_'${N}'.csv'
echo "split texts into sentences"
python src/read_mcscript.py ${output}  ${N}


#######
echo "parse sentence with srl"
python src/allen_srl.py \
  --input_file output/MCScript2.0/${output}\
  --batch_size 32 \
  --output_file output/${output}_frame.csv
  --debug

# ##########
python src/graph_normalize.py ${output}
sort -u output/${output}.raw.en.csv >output/${output}.en.csv 
# wc -l output/${output}.en.csv  

python src/spacy_src/preprocess_kg.py ${output} output/${output}.en.csv output/${output}.lemma.en.csv

########
# echo 'grounding KG2 on KG1 '
# kg='cn'
# python src/kgsrc/ground_swow_on_conceptnet.py\
#   --conceptnet_source_file data/conceptnet.en.csv\
#   --swow_source_file output/${output}.lemma.en.csv\
#   --output_csv_path output/${output}_${kg}_overlap.csv\
#   --output_vocab_path output/${output}_${kg}_vocab.csv\
#   --output_relation_path output/relation.csv\
#   --input_order rhtw\
#   --write_non_overlap\
#   --output_csv_path_non_overlap output/${output}_${kg}_non_overlap.csv\
#   --output_vocab_path_non_overlap  output/${output}_${kg}_non_overlap_vocab.csv\
#   --output_relation_path_non_overlap  output/relation.csv

# #   # --swow_source_file data/swow.en.csv\

# kg='sw'
# python src/kgsrc/ground_swow_on_conceptnet.py\
#   --conceptnet_source_file data/swow.en.csv\
#   --swow_source_file output/${output}.lemma.en.csv\
#   --output_csv_path output/${output}_${kg}_overlap.csv\
#   --output_vocab_path output/${output}_${kg}_vocab.csv\
#   --input_order rhtw\
#   --output_relation_path output/relation.csv\
#   --write_non_overlap\
#   --output_csv_path_non_overlap output/${output}_${kg}_non_overlap.csv\
#   --output_vocab_path_non_overlap  output/${output}_${kg}_non_overlap_vocab.csv\
#   --output_relation_path_non_overlap  output/relation.csv
# ##########






##########
# echo 'print graph statistics'
# python src/kgsrc/graph_statistics.py --dataset mcscript2\
#   --source_file output/${output}.raw.en.csv\
#   --input_order rhtw\
#   --vocab_path output/mcscript2.vocab\
#   --out_network

