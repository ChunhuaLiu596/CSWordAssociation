# Commonsense Knowledge in Word Associations and ConceptNet

This is a Pytorch implementation for our CoNLL 2021 paper: Commonsense Knowledge in Word Associations and ConceptNet [[pdf](https://aclanthology.org/2021.conll-1.38.pdf)]

This paper compares how commonsense knowledge encoded in commonsense knowledge graph **ConceptNet** different a large-scale word association network (**SWOW**, the Small World of Words; [De Deyne., (2019)](https://link.springer.com/article/10.3758/s13428-018-1115-7)).

Code folders:

(1) `commonsense-qa`: apply two knowledge graphs to downstream commonsense-qa tasks.

(2) `intrinsic-comparisions`: compare commonsense knowledge encoded in ConceptNet and SWOW from multiple aspects.

(3) `mcscript-coverage`: compare the commonsense knowledge coverage of MCscript2.0 in ConceptNet and SWOW. 

(4) `learning-generator`: conduct path sampling and then train a path generator, which is used in some KG-augmented models in (1) (e.g., RN)


## Dependencies

- Python >= 3.6
- PyTorch == 1.1
- transformers == 2.8.0
- dgl == 0.3.1 (GPU version)
- networkx == 2.3
- tqdm==4.64.1
- sentencepiece=0.1.99
- nltk==3.6.7
- spacy=2.1.6
- matplotlib=3.3.4
- numpy ==1.19.2

Run the following commands to create a conda environment:

```bash
conda create -n csqa python=3.6 numpy matplotlib ipython
source activate csqa
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu110
pip install numpy==1.19.2 matplotlib==3.3.4 dgl-cu100==0.3.1
pip install transformers==2.8.0 tqdm==4.64.1 networkx==2.5.1 nltk==3.6.7 spacy==2.1.6
python -m spacy download en
pip install sentencepiece==0.1.9
```

## Training a Commonsense QA model
### 0. Go to the commonsense-qa folder:  
```bash
cd commonsense-qa 
```

### 1. Download Data
You can download the data from [here](https://drive.google.com/drive/folders/1D_pLTgwgyEZLOUfnNWK1jNrLP0hhA2e3?usp=sharing).
Download the 'data' foler and put it under 'commonsense-qa'. 

### 2. Preprocess

#### 2.1 Preprocess the datasets or knowledge graphs 
To preprocess the data, use [commonsense-qa/preprocess.py](./commonsense-qa/preprocess.py). For example, preprocess the obqa dataset with the SWOW as the knowledge graph, use

```bash
python preprocess.py swow --run obqa 
```
See [commonsense-qa/scripts/run_preprocess.sh](commonsense-qa/scripts/run_preprocess.sh) for a full list of preprocessing all datasets and knowledge graphs. 

#### 2.2. Using the path generator to connect question-answer entities. This is for RN model or any model that used the PathGenerator Model [(Wang et al., 2020)](https://aclanthology.org/2020.findings-emnlp.369.pdf). 


Path generator can generate a path embedding to connect each question-answer concept pair. Details of training a path generator is introduced later in Step 5. 
The processed path embeddings are stored in './commonsense-qa/saved_models/pretrain_generator'[download](https://drive.google.com/drive/folders/1KzsjwCSTsULQlnYMhWpH807VvgN0llB6?usp=sharing), including:   
```bash
├── csqa
│   ├── path_embedding_cpnet.pickle
│   └── path_embedding_swow.pickle
├── mcscript
│   ├── path_embedding_cpnet.pickle
│   └── path_embedding_swow.pickle
└── obqa
    ├── path_embedding_cpnet.pickle
    └── path_embedding_swow.pickle
```
Alternatively, you can generate them with: 
```bash
./scripts/run_generate.sh
```
Note: 
1. You need to train a path generator using guidelines in step 5 before running the above command. 
2. You need to modify ./config/path_generate.config to specify the dataset and gpu device. 

#### 2.3. TransE embeddings 
We used TransE to train a KG and use the relational embedding matrix for the RN model. 
You can download the trained 'transe_embeddings' from [here](https://drive.google.com/drive/folders/1NWgy_VvnXmHrhQKqy_hfL9BM6IKr89wb?usp=sharing) and put it under the 'commonsense-qa' folder


### 3. Commonsense QA model training
The key componentents of a commonsense QA model include a text encoder and a knowledge graph (KG) encoder. 

We use the pre-trained ALBERT-xxlargev2 (Lan et al., 2020) as our text encoder, which represents the textual information in each question and each answer option. The model is downloaded and put in the folder of './commonsense-qa/cache/albert-xxlarge-v2'. You can change the directory and upate the location of 'cache_dir' in the 'modelling/modeling_encoder.py' file.

We use two KG encoders in our paper: GConAttn and RN. Next, I will use the RN model and the OBQA dataset to show the training commands.

To train a model, specify the config, a knowledge graph name, and the KG model name. For example, to train an RN model with ConceptNet (17 relation types) on the OBQA dataset, you can use the following command:

```bash
bash ./scripts/run_main.sh config/obqa.config cpnet17rel  
```

You can use the following command to train a RN model on SWOW:

```bash
bash ./scripts/run_main.sh config/obqa.config swow rn
```

Training process and final evaluation results would be stored in './saved_models/'


## 5. For training a path generator
This is taken from the [Commonsense-Path-Generator](https://github.com/wangpf3/Commonsense-Path-Generator) repository. 
You can downlaod the data from [here](https://drive.google.com/drive/folders/1D67MdMNe2Dyymzn30gwxpsjeqqsSxKrS?usp=sharing) and put it under learning-generator
```bash
cd learning-generator
cd data
unzip conceptnet.zip
cd ..
python sample_path_rw.py

python sample_path_rw.py --data_dir data/conceptnet/ --output_dir data/sample_path_conceptnet
python sample_path_rw.py --data_dir data/cpnet_base/ --output_dir data/sample_path_cpnet_base
python sample_path_rw.py --data_dir data/swow/ --output_dir data/sample_path_swow
```

After path sampling, shuffle the resulting data './data/sample_path/sample_path.txt'
and then split them into train.txt, dev.txt and test.txt by ratio of 0.9:0.05:0.05 under './data/sample_path/'

```bash
bash split_path_files data/sample_path_conceptnet
bash split_path_files data/sample_path_swow
```

Then you can start to train the path generator by running

```bash
# the first arg is for specifying which gpu to use
./run.sh $gpu_device
```

After path sampling, shuffle the resulting data './data/sample_path/sample_path.txt'
and then split them into train.txt, dev.txt and test.txt by ratio of 0.9:0.05:0.05 under './data/sample_path/'

Then you can start to train the path generator by running
```bash
# the first arg is for specifying which gpu to use
./run.sh $gpu_device
```

The checkpoint of the path generator would be stored in './checkpoints/model.ckpt'.
Move it to '../commonsense-qa/saved_models/pretrain_generator'.
So far, we are done with training the generator. After this, you can use calc_path_embedding.py to calculate path embeddings for a specified dataset.

  
## Citation

```
@inproceedings{liu-etal-2021-commonsense,
    title = "Commonsense Knowledge in Word Associations and {C}oncept{N}et",
    author = "Liu, Chunhua  and
      Cohn, Trevor  and
      Frermann, Lea",
    editor = "Bisazza, Arianna  and
      Abend, Omri",
    booktitle = "Proceedings of the 25th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.conll-1.38",
    doi = "10.18653/v1/2021.conll-1.38",
    pages = "481--495"
}
```

## Acknolwedgemnt

Our commonsense-qa models are largly based on the [MHGRN](https://github.com/INK-USC/MHGRN) ([Feng et al., 2020](https://aclanthology.org/2020.emnlp-main.99.pdf)) and [Commonsense-Path-Generator](https://github.com/wangpf3/Commonsense-Path-Generator) repository. We thank the authors for providing the resources and answering questions.
