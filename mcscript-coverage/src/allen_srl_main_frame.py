from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from contextlib import ExitStack
import re
import argparse
import json
import pandas as pd
import allennlp_models.tagging
from itertools import islice
from tqdm import tqdm 
tqdm.pandas()

# predictor.predict(
#     sentence="Did Uriah honestly think he could beat the game in under three hours?."
# )
# revised based on https://github.com/masrb/Semantic-Role-Labeling-allenNLP-/blob/master/allen_srl.py

'''
# usage: 
python --input_file --output_file --batch_size 20 --parse_frame True 
'''


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=argparse.FileType('r'), help='path to input file')
    parser.add_argument('--output_file', type=argparse.FileType('w'), help='path to output file')
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for processing')
    parser.add_argument('--parse_frame', action='store_true', help='whether parse frame with SRL')
    parser.add_argument('--debug', action='store_true', help='')

    args = parser.parse_args()

    return args

def get_predictor():
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    return predictor


def next_n_lines(file_opened, N):
    # return [x.strip() for x in islice(file_opened, N)]
    while True:
        lines = [x.strip() for x in islice(file_opened, N)]

        if not lines:
            break
        yield lines


def run_predictor_sentence(predictor, sentences, batch_size,  output_file=None, print_to_console=None):
    '''
    sentence prediction {'verbs': [{'verb': 'bought', 'description': '[ARG0: Anna] [V: bought] [ARG1: a car] [ARG2: from Allen]', 'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'I-ARG2']}], 'words': ['Anna', 'bought', 'a', 'car', 'from', 'Allen']}
    '''
    relations = {'ARG0': 'ARG0V', 
                 'ARG1': 'ARG1V',
                 'ARG01': 'ARG01'}

    frames = set()

    def _run_predictor_batch(batch_data):
        if len(batch_data) == 1:
            # output  = predictor.predict(sentence)
            result = predictor.predict_json(batch_data[0])
            results = [result]
        else:
            results = predictor.predict_batch_json(batch_data)
        return results

    for i in range(0, len(sentences), batch_size):
        batch_sentence  = sentences[i:i+batch_size] 
        batch_data = [{'sentence': sentence} for sentence in batch_sentence]
        results = _run_predictor_batch(batch_data)

        for sentence, output in zip(batch_data, results): 
            string_output = predictor.dump_line(output)
           
            if len(output['verbs']) ==0: continue

            description = output['verbs'][0]['description']

            frame = re.findall(r'\[([^\[\]]+)\]', description)

            role = ('ARG0', 'ARG1', 'V') 
            cur_role = {}
            for x in frame:
                y = x.split(": ") #split by ": " is not 100 accurate, 8:30
                if len(y)<2: 
                    print("bad case ", y)
                    continue
                # if len(y)>2:
                    # print(y)

                k, v = y[0], y[1]
                if k in role:
                    cur_role[k] = v

            if 'V' in cur_role:
                for key in ['ARG0', 'ARG1']:
                    if key in cur_role:
                        triple = (relations.get(key), cur_role[key], cur_role['V'])
                        frames.add(triple)

                if all( key in cur_role for key in ['ARG0', 'ARG1']):
                    triple = (relations.get("ARG01"), cur_role['ARG0'], cur_role['ARG1'])
                    frames.add(triple)

            if print_to_console:
                print("input: ", sentence)
                print("batch data prediction: ", string_output)
                for k,v in cur_role.items():
                    print(k, v)
                print("")
    if output_file:
        df = pd.DataFrame(list(frames), columns =['rel', 'head', 'tail'])
        # output_file.write(string_output)
        df.to_csv(output_file, index=False)
        print(f"save {output_file} {len(df.index)} lines")
    return frames 

def test_example():
    # batch_data = [{'sentence': 'Which NFL team represented the AFC at Super Bowl 50?'},{'sentence': 'Where did Super Bowl 50 take place?'},{'sentence': 'Which NFL team won Super Bowl 50?'}]
    # batch_data = ['Which NFL team represented the AFC at Super Bowl 50?',  'Where did Super Bowl 50 take place?','Which NFL team won Super Bowl 50?']

    sentence= "Anna bought a car from Allen"
    # run_predictor_sentence(sentence)	

def frame_to_triple(frames):
    from graph_normalize import reconstruct_triple

    triple = set()
    triple_lemma = set()
    for (rel, head, tail) in frames:
        # for frame in frames:
        # rel, head, tail = frame 
        cur_triple, cur_triple_lemma = reconstruct_triple(rel, head, tail)
        triple.update(cur_triple)
        triple_lemma.update(cur_triple_lemma)

    return pd.Series([triple, triple_lemma])

def run(predictor,
        df,
        batch_size,
        output_file,
        print_to_console
        ):

    df['frame'] = df['sent'].progress_apply(lambda x: run_predictor_sentence(predictor, eval(x), batch_size))
    print(df.frame)
    df['triple'] = df['frame'].progress_apply(lambda x: frame_to_triple(x))

    df.to_csv(output_file, index=False)
    print(f"save {output_file} {len(df.index)} lines")
    print(df.describe()) 
    # run_predictor_sentence(predictor, input_sentences, batch_size, output_file, print_to_console)

def convert_parsed_frame(df, output_path):
    '''
    convert  frame to triple (both orth and lemma)
    '''
    # if 'triple' not in df.columns:
        # df['triple'] = df['frame'].progress_apply(lambda x: frame_to_triple(x))

    df[['triple', 'triple_lemma']] = df['frame'].progress_apply(lambda x: frame_to_triple(eval(x)))
    df.to_csv(output_path)
    print(f"save {output_path} {len(df.index)} lines")
    return df 

def read_input_file(path, debug=False):
    df = pd.read_csv(path)
    if debug:
        # return df['sent'][:20]
        return df[:5]
    print(f"read {path} {len(df.index)} lines") 
    # return df['sent']
    return df


def main():
    args = get_arguments()
    predictor = get_predictor()
    output_file = None
    print_to_console = False

    df = read_input_file(args.input_file, args.debug)

    print(args)
    with ExitStack() as stack:
          
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  
        else:
            print_to_console = True

        if args.parse_frame:
            run(predictor,
                df,
                args.batch_size, 
                output_file,
                print_to_console
                )
        else:
            convert_parsed_frame(df, output_file)

if __name__ == '__main__':
    main()

    