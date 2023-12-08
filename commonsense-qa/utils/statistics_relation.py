
load_2hop_relational_paths

def cal_2hop_rel_emb(rel_emb):
    n_rel = rel_emb.shape[0]
    u, v = np.meshgrid(np.arange(n_rel), np.arange(n_rel))
    expanded = rel_emb[v.reshape(-1)] + rel_emb[u.reshape(-1)] #posy: 2hop relemb = rel1 + rel2
    return np.concatenate([rel_emb, expanded], 0)


def calc_rel_emb()
    relation_num, relation_dim = 210, 100
    rel_emb = nn.Embedding(relation_num, relation_dim)
    rel_ids= [36, 17]

    rel_embed = rel_emb(rel_ids)
    n_1hop_rel = int(np.sqrt(self.relation_num)) # 14

    rel_ids = rel_ids.view(bs * sl) #[36, 7] 
    twohop_mask = rel_ids >= n_1hop_rel   ## rel_ids=36 （7*2+7*2+8) 
    #twohop_mask=[1, 0]

    twohop_rel = rel_ids[twohop_mask] - n_1hop_rel #filter out one hop rel?
    r1, r2 = twohop_rel // n_1hop_rel, twohop_rel % n_1hop_rel

    assert (r1 >= 0).all() and (r2 >= 0).all() and (r1 < n_1hop_rel).all() and (r2 < n_1hop_rel).all()
    rel_embed = rel_embed.view(bs * sl, -1)
    rel_embed[twohop_mask] = torch.mul(rel_emb(r1), rel_emb(r2)) #posy
    rel_embed = rel_embed.view(bs, sl, -1)

def load_rel_emb(args):
    rel_emb = np.load(args.rel_emb_path)
    rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    rel_emb = cal_2hop_rel_emb(rel_emb)
    rel_emb = torch.tensor(rel_emb)
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1) # (14*14+14, 100)
    rerutn rel_emb, relation_num, relation_dim

