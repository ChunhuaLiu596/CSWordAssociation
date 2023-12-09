
#get the post tags for CN and SW
python -u src/pos_entities.py --debug $1 --reload $2
python -u src/pos_swow_cue.py
