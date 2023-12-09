#/bin/bash


output='mcscript2.csv'

#######
# echo "split texts into sentences"
python src/read_mcscript.py ${output} 


# #######
# echo "parse sentence with srl"
# python src/allen_srl.py \
#   --input_file output/mcscript2/${output}\
#   --batch_size 32 \
#   --output_file output/${output}_frame.csv\
#   --parse_frame\

######debug #######
python src/allen_srl.py \
  --input_file output/mcscript2/${output}\
  --batch_size 32 \
  --output_file output/${output}_frame_debug.csv\
  --parse_frame\
  --debug
# ######debug #######


# echo "convert parsed frames to triples"
# python src/allen_srl.py \
  # --input_file output/${output}_frame.csv\
  # --batch_size 32 \
  # --output_file output/${output}_frame_lemma.csv\
  # --debug

##### 
# python src/graph_normalize.py ${output}
# sort -u output/${output}.raw.en.csv >output/${output}.en.csv 
# wc -l output/${output}.en.csv  
