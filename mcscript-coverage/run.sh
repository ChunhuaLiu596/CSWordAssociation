

# python src/read_mcscript.py 
# python src/allen_srl.py \
#   --input_file output/MCScript2.0/mcscript2.csv\
#   --batch_size 32 \
#   --output_file output/mcscript2_frame.csv
  # --debug


##########
# echo 'print graph statistics'
# python src/kgsrc/graph_statistics.py --dataset mcscript2\
  # --source_file output/mcscript2.raw.en.csv\
  # --input_order rhtw\
  # --vocab_path output/mcscript2.vocab\
  # --out_network


# ##########
# python src/graph_normalize.py
# sort -u output/mcscript2.raw.en.csv >output/mcscript2.en.csv 
# echo wc -l output/mcscript2.en.csv  

########
echo 'grounding KG2 on KG1 '
kg='cn'
python src/kgsrc/ground_swow_on_conceptnet.py\
  --conceptnet_source_file data/conceptnet.en.csv\
  --swow_source_file output/mcscript2.lemma.en.csv\
  --output_csv_path output/mcscript2_${kg}_overlap.csv\
  --output_vocab_path output/mcscript2_${kg}_vocab.csv\
  --output_relation_path output/relation.csv\
  --input_order rhtw\
  --write_non_overlap\
  --output_csv_path_non_overlap output/mcscript2_${kg}_non_overlap.csv\
  --output_vocab_path_non_overlap  output/mcscript2_${kg}_non_overlap_vocab.csv\
  --output_relation_path_non_overlap  output/relation.csv

  # --swow_source_file data/swow.en.csv\

kg='sw'
python src/kgsrc/ground_swow_on_conceptnet.py\
  --conceptnet_source_file data/swow.en.csv\
  --swow_source_file output/mcscript2.lemma.en.csv\
  --output_csv_path output/mcscript2_${kg}_overlap.csv\
  --output_vocab_path output/mcscript2_${kg}_vocab.csv\
  --input_order rhtw\
  --output_relation_path output/relation.csv\
  --write_non_overlap\
  --output_csv_path_non_overlap output/mcscript2_${kg}_non_overlap.csv\
  --output_vocab_path_non_overlap  output/mcscript2_${kg}_non_overlap_vocab.csv\
  --output_relation_path_non_overlap  output/relation.csv
##########



# python src/spacy_src/preprocess_kg.py mcscript2 output/mcscript2.en.csv output/mcscript2.lemma.en.csv

# After preprocessing, you can go to src/shortest_path_length.ipynb to get the path lengths for grounded MCScripts graphs
