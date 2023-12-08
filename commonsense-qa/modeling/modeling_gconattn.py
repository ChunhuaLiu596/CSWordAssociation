import torch.nn as nn
from utils.data_utils import *
from utils.layers import *
from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS

class GconAttn(nn.Module):
    def __init__(self, concept_num, concept_dim, concept_in_dim, pretrained_concept_emb, freeze_ent_emb, hidden_dim, dropout):
        super().__init__()
        self.concept_emb = CustomizedEmbedding(concept_num=concept_num, concept_out_dim=concept_dim, concept_in_dim=concept_in_dim,
                                                pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb, use_contextualized=False)
        self.hidden_dim = hidden_dim

        self.sim = DotProductSimilarity()
        self.attention = MatrixAttention(self.sim)
        self.MLP = MLP(input_size=4 * concept_dim, hidden_size=hidden_dim, output_size=hidden_dim, num_layers=1, dropout=dropout, batch_norm=False, layer_norm=True)
        self.max_pool = MaxPoolLayer()
        self.mean_pool = MeanPoolLayer()

        self.dropout = nn.Dropout(dropout)

        self.kgvec_dim = 4 * hidden_dim
        self.Agg = MLP(input_size=2 * self.kgvec_dim, hidden_size=hidden_dim, output_size=self.kgvec_dim, num_layers=1, dropout=dropout, batch_norm=False, layer_norm=True)

    def forward(self, p_id, q_id, a_id, p_num, q_num, a_num):
        """
        q_id: (nbz, seq_len)
        a_id: (nbz, seq_len)
        q_num: (nbz,)
        a_num: (nbz,)
        """

        bz, sl = q_id.size()
        p_num = torch.max(p_num, torch.tensor(1).to(p_num.device)).unsqueeze(1)
        q_num = torch.max(q_num, torch.tensor(1).to(q_num.device)).unsqueeze(1)
        a_num = torch.max(a_num, torch.tensor(1).to(a_num.device)).unsqueeze(1)

        # print(f"p_num: {p_num}")
        # print(f"q_num: {q_num}")
        # print(f"a_num: {a_num}")

        pmask = torch.arange(sl, device=p_id.device) >= p_num  # (nbz, sl)
        qmask = torch.arange(sl, device=q_id.device) >= q_num  # (nbz, sl)
        amask = torch.arange(sl, device=a_id.device) >= a_num  # (nbz, sl)
        # print(f"pmask: {pmask}")
        # print(f"qmask: {qmask}")
        # print(f"amask: {amask}")

        qa_vecs = self.inference(q_id, a_id, qmask, amask, q_num, a_num)
        # print(f"qa_vecs: {qa_vecs}")
        pa_vecs = self.inference(p_id, a_id, pmask, amask, p_num, a_num)
        # print(f"pa_vecs: {pa_vecs}")

        vecs = self.dropout(torch.cat((qa_vecs, pa_vecs), dim=-1))
        kg_vecs = self.Agg(vecs)
        # print(f"kg_vecs: {kg_vecs}")
        return kg_vecs

    def inference(self, q_id, a_id, qmask, amask, q_num, a_num):
        mask = qmask.unsqueeze(2) | amask.unsqueeze(1)  # (nbz, sl, sl)
        # print(f"mask: {mask}")
        # print(f"q_id: {q_id}")
        # print(f"a_id: {a_id}")
        q = self.concept_emb(q_id)  # (nbz, sl, cpt_dim)
        # print(f"q: {q}")
        a = self.concept_emb(a_id)  # (nbz, sl, cpt_dim)
        # print(f"a: {a}")

        attn = self.attention(q, a)  # (nbz, sl, sl)
        # print(f"attn: {attn}")
        q2a = masked_softmax(attn, mask, dim=-1)  # (nbz, sl, sl)
        a2q = masked_softmax(attn, mask, dim=0)  # (nbz, sl, sl)

        # print(f"q2a: {q2a}")
        # print(f"a2q: {a2q}")
        beta = (q2a.unsqueeze(3) * a.unsqueeze(1)).sum(2)  # (nbz, sl, cpt_dim), unsqueeze dim of a, sum over dim of a
        alpha = (a2q.unsqueeze(3) * q.unsqueeze(2)).sum(1)  # (nbz, sl, cpt_dim), unsqueeze dim of q, sum over dim of q

        # print(f"beta: {beta}")
        # print(f"alpha: {alpha}")
        # ----- original  (a - beta) -> should be (q - beta) ----------
        # qm = self.MLP(torch.cat((a, beta, a - beta, a * beta), dim=-1))  # (nbz, sl, out_dim)
        # am = self.MLP(torch.cat((q, alpha, q - alpha, q * alpha), dim=-1))  # (nbz, sl, out_dim)
        # ------- revised ----------
        qm = self.MLP(torch.cat((q, beta, q - beta, q * beta), dim=-1))  # (nbz, sl, out_dim)
        am = self.MLP(torch.cat((a, alpha, a - alpha, a * alpha), dim=-1))  # (nbz, sl, out_dim)

        # print(f"beta: {qm}")
        # print(f"alpha: {am}")

        q_mean = self.mean_pool(qm, q_num.squeeze(1))  # (nbz, out_dim)
        q_max = self.max_pool(qm, q_num.squeeze(1))
        a_mean = self.mean_pool(am, a_num.squeeze(1))
        a_max = self.max_pool(am, a_num.squeeze(1))

        # print(f"q_mean: {q_mean}")
        # print(f"q_max: {q_max}")
        # print(f"a_mean: {a_mean}")
        # print(f"a_max: {a_max}")
        vecs = torch.cat((q_mean, q_max, a_mean, a_max), dim=-1)
        # logits = self.hidd2out(torch.cat((q_mean, q_max, a_mean, a_max, s), dim=-1))
        return vecs

class LMGconAttn(nn.Module):
    def __init__(self, model_name, from_checkpoint, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, hidden_dim, dropout, ablation=None, encoder_config={}, label_dim=1, lm_sent_pool='cls'):
        super().__init__()
        self.model_name = model_name
        self.ablation = ablation
        self.encoder = TextEncoder(model_name, from_checkpoint=from_checkpoint, sent_pool=lm_sent_pool, **encoder_config)
        self.decoder = GconAttn(concept_num=concept_num, concept_dim=concept_dim, concept_in_dim=concept_in_dim,
                                freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=pretrained_concept_emb,
                                hidden_dim=hidden_dim, dropout=dropout)

        self.dropout= nn.Dropout(dropout)
        if self.ablation == 'kg_only':
            self.hidd2out = nn.Linear(self.decoder.kgvec_dim, 1)
        else:
            self.hidd2out = nn.Linear( self.encoder.sent_dim +self.decoder.kgvec_dim, 1)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        *lm_inputs, pc, qc, ac, pc_len, qc_len, ac_len = inputs

        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        kg_vecs = self.decoder(p_id=pc, q_id=qc, a_id=ac, p_num=pc_len, q_num=qc_len, a_num=ac_len)

        if self.ablation == 'kg_only':
            kg_sent_vecs = self.dropout(kg_vecs)
        else:
            kg_sent_vecs = self.dropout(torch.cat((kg_vecs, sent_vecs), dim=-1))
        logits = self.hidd2out(kg_sent_vecs).view(bs, nc)
        # print(f"logits1: {logits}")
        return logits, None  # for lstm encoder, attn=None

class KGAttn(nn.Module):
    def __init__(self, model_name, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, hidden_dim, dropout, ablation=None, encoder_config={}):
        super().__init__()
        self.model_name = model_name
        self.encoder = GconAttn(concept_num=concept_num, concept_dim=concept_dim, concept_in_dim=concept_in_dim,
                                freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=pretrained_concept_emb,
                                hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = nn.Linear(self.encoder.kgvec_dim, 1)
        self.dropout= nn.Dropout(dropout)
    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        *lm_inputs, qc, ac, qc_len, ac_len = inputs
        kg_vecs = self.encoder(q_id=qc, a_id=ac, q_num=qc_len, a_num=ac_len)
        logits = self.decoder(self.dropout(kg_vecs)).view(bs, nc)
        return logits, None  # for lstm encoder, attn=None


class GconAttnDataLoader(object):
    def __init__(self, train_statement_path: str, train_concept_jsonl: str, dev_statement_path: str,
                 dev_concept_jsonl: str, test_statement_path: str, test_concept_jsonl: str,
                 concept2id_path: str, batch_size, eval_batch_size, device, model_name=None,
                 max_cpt_num=20, max_seq_length=128, is_inhouse=True, inhouse_train_qids_path=None,
                 subsample=1.0, format=[], pretrained_concept_emb=None, text_only=False, concept2deg_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.is_inhouse = is_inhouse
        self.max_cpt_num = max_cpt_num
        self.vocab = None
        self.pretrained_concept_emb = pretrained_concept_emb

        # model_type = MODEL_NAME_TO_CLASS.get(model_name, 'lstm') # if doesn't retrieve model_name, then use lstm 
        model_type = MODEL_NAME_TO_CLASS.get(model_name)  
        self.train_qids, self.train_labels, *self.train_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)
        self.num_choice = None
        if test_statement_path is not None:
            self.test_qids, self.test_labels, *self.test_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)

        self.num_choice = self.train_data[0].size(1)
        if not text_only:
            self._load_concept_idx(concept2id_path)
            if concept2deg_path is not None:
                self._load_concept_degree(concept2deg_path)
    
            self.train_data += self._load_concepts(train_concept_jsonl)
            self.dev_data += self._load_concepts(dev_concept_jsonl)
            if test_statement_path is not None:
                self.test_data += self._load_concepts(test_concept_jsonl)

        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
        assert all(len(self.dev_qids) == x.size(0) for x in [self.dev_labels] + self.dev_data)
        # if test_statement_path is not None:  ## posy: open later
            # assert all(len(self.test_qids) == x.size(0) for x in [self.test_labels] + self.test_data)
        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_data = [x[:n_train] for x in self.train_data]
                assert all(len(self.train_qids) == x.size(0) for x in [self.train_labels] + self.train_data)
            assert self.train_size() == n_train

    def __getitem__(self, index):
        raise NotImplementedError()

    def _load_concept_idx(self, concept_list):
        with open(concept_list, 'r', encoding='utf8') as fin:
            id2concept = [w.strip() for w in fin]
        self.concept2id = {w: i for i, w in enumerate(id2concept)}
        self.id2concept = id2concept

    def _load_concept_degree(self, concept_list):
        with open(concept_list, 'r', encoding='utf8') as fin:
            concept2deg = [w.strip().split("\t") for w in fin]

        self.concept2deg = {x[0]: int(x[1]) for x in concept2deg}
        assert len(self.concept2deg.keys()) == len(self.concept2id.keys())

    def _load_concepts(self, concept_json):

        with open(concept_json, 'r') as fin:
            concept_data = [json.loads(line) for line in fin]
        n = len(concept_data)
        pc, qc, ac = [], [], []
        pc_len, qc_len, ac_len = [], [], []
        recall_qac = 0
        grounded_c = list()
        for data in tqdm(concept_data, total=n, desc='loading concepts'):
            # leave index 0 for padding
            #cur_qc = [self.concept2id[x] + 1 for x in data['qc']][:self.max_cpt_num]
            #cur_ac = [self.concept2id[x] + 1 for x in data['ac']][:self.max_cpt_num]
            cur_pc = [self.concept2id[x] for x in data['pc']][:self.max_cpt_num]
            cur_qc = [self.concept2id[x] for x in data['qc']][:self.max_cpt_num]
            cur_ac = [self.concept2id[x] for x in data['ac']][:self.max_cpt_num]

            grounded_c.extend([x for x in data['qc']][:self.max_cpt_num])
            grounded_c.extend([x for x in data['ac']][:self.max_cpt_num])
            if len(cur_qc) + len(cur_ac) >=2:
                recall_qac +=1

            pc.append(cur_pc + [0] * (self.max_cpt_num - len(cur_pc)))
            qc.append(cur_qc + [0] * (self.max_cpt_num - len(cur_qc)))
            ac.append(cur_ac + [0] * (self.max_cpt_num - len(cur_ac)))

            # print("{} | {}".format([x for x in data['ac']][:self.max_cpt_num], cur_ac_deg))

            assert len(qc[-1]) == len(ac[-1]) == self.max_cpt_num
            pc_len.append(len(cur_pc))
            qc_len.append(len(cur_qc))
            ac_len.append(len(cur_ac))

        print('avg_num_qc = {}'.format(sum(qc_len) / float(len(qc_len))))
        print('avg_num_ac = {}'.format(sum(ac_len) / float(len(ac_len))))
        print('grounded concepts num = {} (question + answer)'.format(len(set(grounded_c))))
        print('concept recall rate = {} (more than one grounded concepts for each qa-pair)'.format( recall_qac / n))
        pc, qc, ac = [torch.tensor(np.array(x).reshape((-1, self.num_choice, self.max_cpt_num))) for x in [pc, qc, ac]]
        pc_len, qc_len, ac_len = [torch.tensor(np.array(x).reshape((-1, self.num_choice))) for x in [pc_len, qc_len, ac_len]]

        return pc, qc, ac, pc_len, qc_len, ac_len

    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    def dev_size(self):
        return len(self.dev_qids)

    def test_size(self):
        return self.inhouse_test_indexes.size(0) if self.is_inhouse else len(self.test_qids)

    def _to_device(self, obj):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item) for item in obj]
        else:
            return obj.to(self.device)

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return BatchGenerator(self.device, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors=self.train_data)

    def dev(self):
        return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors=self.dev_data)

    def test(self):
        if self.is_inhouse:
            return BatchGenerator(self.device, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors=self.train_data)
        else:
            return BatchGenerator(self.device, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors=self.test_data)


#----- back up codes -----

class TextKGAttn(nn.Module):
    def __init__(self, model_name, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, hidden_dim, dropout, ablation=None, encoder_config={}, label_dim=1):
        super().__init__()
        self.model_name = model_name

        pretrained_emb_or_path = encoder_config['pretrained_emb_or_path']
        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                emb = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb_num, emb_dim = emb.size(0), emb.size(1)

        hidden_dim_encoder = encoder_config['hidden_size']
        self.encoder = GconAttn(concept_num=emb_num, concept_dim=emb_dim, concept_in_dim=emb_dim,
                                freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=emb,
                                hidden_dim= hidden_dim_encoder, dropout=dropout)

        self.decoder = GconAttn(concept_num=concept_num, concept_dim=concept_dim, concept_in_dim=concept_in_dim,
                                freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=pretrained_concept_emb,
                                hidden_dim=hidden_dim, dropout=dropout)

        self.hidd = nn.Linear(4 * hidden_dim_encoder + 4 * hidden_dim, hidden_dim_encoder+hidden_dim)
        self.activation= nn.ReLU()
        self.hidd2out = nn.Linear(hidden_dim_encoder + hidden_dim, 1)
        self.hidd_dropout = nn.Dropout(dropout)
        # self.hidd2out = nn.Linear(4 * hidden_dim + 4 * hidden_dim_encoder, 1)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        qt, at, qt_len, at_len, qc, ac, qc_len, ac_len  = inputs
        text_vecs, attn = self.encoder(qt, at, qt_len, at_len)

        kg_vecs, attn = self.decoder(q_id=qc, a_id=ac, q_num=qc_len, a_num=ac_len)
        hidden = self.activation(self.hidd(self.hidd_dropout(torch.cat((kg_vecs, text_vecs), dim=-1))))
        logits = self.hidd2out(hidden).view(bs, nc)
        # logits = self.hidd2out(torch.cat((kg_vecs, text_vecs), dim=-1)).view(bs, nc)
        return logits, attn  #attn=None


class TextAttn(nn.Module):
    def __init__(self, model_name, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, hidden_dim, dropout, ablation=None, encoder_config={}, label_dim=1):
        super().__init__()
        self.model_name = model_name
        pretrained_emb_or_path = encoder_config['pretrained_emb_or_path']
        if pretrained_emb_or_path is not None:
            if isinstance(pretrained_emb_or_path, str):  # load pretrained embedding from a .npy file
                emb = torch.tensor(np.load(pretrained_emb_or_path), dtype=torch.float)
            emb_num, emb_dim = emb.size(0), emb.size(1)

        self.encoder = GconAttn(concept_num=emb_num, concept_dim=emb_dim, concept_in_dim=emb_dim,
                                freeze_ent_emb=freeze_ent_emb, pretrained_concept_emb=emb,
                                hidden_dim=hidden_dim, dropout=dropout)

        self.hidd2out = nn.Linear(4 * hidden_dim, 1)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension

        qt, at, qt_len, at_len, *concept_inputs = inputs
        sent_vecs, _ = self.encoder( qt, at, qt_len, at_len)
        logits = self.hidd2out(sent_vecs).view(bs, nc)
        return logits #, None  # for lstm encoder, attn=None



class TextModel(nn.Module):
    def __init__(self, model_name, concept_num, concept_dim, concept_in_dim, freeze_ent_emb, pretrained_concept_emb, hidden_dim, dropout, ablation=None, encoder_config={}, label_dim=1):
        super().__init__()
        self.model_name = model_name
        self.encoder = TextEncoder(model_name, **encoder_config)
        self.decoder = nn.Linear(self.encoder.sent_dim, 1)

    def forward(self, *inputs, layer_id=-1):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs]  # merge the batch dimension and the num_choice dimension
        # print(inputs)
        if self.model_name in ("qacompare", "lstmcompare"):
            *lm_inputs, qc, ac, qc_len, ac_len = inputs
            sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        else:
            # *lm_inputs = inputs

            sent_vecs, all_hidden_states = self.encoder(*inputs, layer_id=layer_id)

        logits = self.decoder(sent_vecs).view(bs, nc)
        return logits #, None  # for lstm encoder, attn=None


