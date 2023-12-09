from pos_entities import load_concepts, get_pos_spacy 


def main():
    concepts_cn = load_concepts("data/swow/swow_cues.csv")
    pos_cn, pos_cn_norm, phrase_cn = get_pos_spacy(concepts_cn, "swow_cues", concept_pos_path="data/swow/swow_cue_pos.csv", debug=False)
    print(pos_cn)
    print(pos_cn_norm)


if __name__=='__main__':
    main()
