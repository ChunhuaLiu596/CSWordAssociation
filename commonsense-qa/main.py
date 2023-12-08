import os,sys
import random
import json
from multiprocessing import cpu_count

from transformers import *

from modeling.modeling_lm import LMForMultipleChoice, LMDataLoader
from modeling.modeling_rn_pg import *
from modeling.modeling_gconattn import *
from utils.optimization_utils import OPTIMIZER_CLASSES
from utils.parser_utils import *
from utils.relpath_utils import *
from utils.datasets import *
import warnings 
warnings.filterwarnings("ignore")

def get_node_feature_encoder(encoder_name):
    return encoder_name.replace('-cased', '-uncased')

def get_merged_relations(kg_name):
    print(f"kg_name: {kg_name}")
    if kg_name in ('cpnet'):
        from utils.conceptnet import merged_relations
    elif kg_name in ('cpnet7rel'):
        from utils.conceptnet import merged_relations_7rel as merged_relations
    elif kg_name in ('cpnet1rel'):
        from utils.conceptnet import merged_relations_1rel as merged_relations
    elif kg_name in ('swow'):
        from utils.swow import merged_relations
    elif kg_name in ('swow1rel'):
        from utils.swow import merged_relations_1rel as merged_relations
    elif kg_name in ('cpnet_swow'):
        from utils.conceptnet_swow import merged_relations
    return merged_relations


def cal_2hop_rel_emb(rel_emb):
    n_rel = rel_emb.shape[0]
    u, v = np.meshgrid(np.arange(n_rel), np.arange(n_rel))
    expanded = rel_emb[v.reshape(-1)] + rel_emb[u.reshape(-1)] #posy: 2hop relemb = rel1 + rel2
    return np.concatenate([rel_emb, expanded], 0)


def evaluate_accuracy(eval_set, model):
    n_samples, n_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            n_correct += (logits.argmax(1) == labels).sum().item()
            n_samples += labels.size(0)
    return n_correct / n_samples

def pred_to_file(eval_set, model, output_path):
    model.eval()
    fw = open(output_path, 'w')
    with torch.no_grad():
        for qids, labels, *input_data in eval_set:
            logits, _ = model(*input_data)
            for qid, pred_label in zip(qids, logits.argmax(1)):
                fw.write('{},{}\n'.format(qid, chr(ord('A') + pred_label.item())))
    fw.close()

def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/', help='model output directory')
    parser.add_argument('--gen_dir', type=str)
    parser.add_argument('--gen_id', type=int)

    # for finding relation paths
    parser.add_argument('--cpnet_vocab_path', default=f'./data/{args.kg_name}/concept.txt')
    parser.add_argument('--cpnet_graph_path', default=f'./data/{args.kg_name}/conceptnet.en.pruned.graph')
    parser.add_argument('--graph_only', type=bool_flag, default=False, help='use concept embeddings from kg only for training')
    parser.add_argument('--path_embedding_path', default=f'./path_embeddings/{args.dataset}/path_embedding_{args.kg_name}.pickle')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')

    # data
    parser.add_argument('--train_rel_paths', default=f'./data/{args.dataset}/{args.kg_name}/paths/train.relpath.2hop.jsonl')
    parser.add_argument('--dev_rel_paths', default=f'./data/{args.dataset}/{args.kg_name}/paths/dev.relpath.2hop.jsonl')
    parser.add_argument('--test_rel_paths', default=f'./data/{args.dataset}/{args.kg_name}/paths/test.relpath.2hop.jsonl')
    parser.add_argument('--train_adj', default=f'./data/{args.dataset}/{args.kg_name}/graph/train.graph.adj.pk')
    parser.add_argument('--dev_adj', default=f'./data/{args.dataset}/{args.kg_name}/graph/dev.graph.adj.pk')
    parser.add_argument('--test_adj', default=f'./data/{args.dataset}/{args.kg_name}/graph/test.graph.adj.pk')
    parser.add_argument('--train_concepts', default=f'./data/{args.dataset}/{args.kg_name}/grounded/train.grounded.jsonl')
    parser.add_argument('--dev_concepts', default=f'./data/{args.dataset}/{args.kg_name}/grounded/dev.grounded.jsonl')
    parser.add_argument('--test_concepts', default=f'./data/{args.dataset}/{args.kg_name}/grounded/test.grounded.jsonl')
    parser.add_argument('--train_node_features', default=f'./data/{args.dataset}/features/train.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--dev_node_features', default=f'./data/{args.dataset}/features/dev.{get_node_feature_encoder(args.encoder)}.features.pk')
    parser.add_argument('--test_node_features', default=f'./data/{args.dataset}/features/test.{get_node_feature_encoder(args.encoder)}.features.pk')

    parser.add_argument('--node_feature_type', choices=['full', 'cls', 'mention'])
    parser.add_argument('--use_cache', default=True, type=bool_flag, nargs='?', const=True, help='use cached data to accelerate data loading')
    parser.add_argument('--max_tuple_num', default=100, type=int)
    parser.add_argument('--text_only', type=bool_flag, default=False, help='use text encoder only for training')

    # model architecture
    parser.add_argument('--ablation', default='att_pool', choices=['None', 'no_kg', 'no_2hop', 'no_1hop', 'no_qa', 'no_rel',
                                                             'mrloss', 'fixrel', 'fakerel', 'no_factor_mul', 'no_2hop_qa',
                                                             'randomrel', 'encode_qas', 'multihead_pool', 'att_pool', 'kg_only'], nargs='?', const=None, help='run ablation test')
    parser.add_argument('--kg_model', default='pg_full', choices=['None', 'pg_full', 'pg_global', 'rn', 'gconattn'], nargs='?', const=None, help='choose kg infusion model')                                                            
    parser.add_argument('--encoder_type', default='PLM', help='pre-trained language model')
    parser.add_argument('--relation_types', default=17, choices=[17, 7, 2], help='relation types in of the knowledge graph') 

    parser.add_argument('--att_head_num', default=2, type=int, help='number of attention heads')
    parser.add_argument('--mlp_dim', default=128, type=int, help='number of MLP hidden units')
    parser.add_argument('--mlp_layer_num', default=2, type=int, help='number of MLP layers')
    parser.add_argument('--fc_dim', default=128, type=int, help='number of FC hidden units')
    parser.add_argument('--fc_layer_num', default=0, type=int, help='number of FC layers')
    parser.add_argument('--freeze_ent_emb', default=True, type=bool_flag, nargs='?', const=True, help='freeze entity embedding layer')
    parser.add_argument('--init_range', default=0.02, type=float, help='stddev when initializing with normal distribution')
    parser.add_argument('--emb_scale', default=1.0, type=float, help='scale pretrained embeddings')
    parser.add_argument('--decoder_hidden_dim', default=300, type=int, help='number of LSTM hidden units')

    # regularization
    parser.add_argument('--dropoutm', type=float, default=0.3, help='dropout for mlp hidden units (0 = no dropout')

    # optimization
    parser.add_argument('-dlr', '--decoder_lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=0, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--grad_step', default=1, type=int)

    parser.add_argument('--subsample', default=1.0, type=float)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.debug:
        parser.set_defaults(batch_size=2, log_interval=1, eval_interval=5)

    # set ablation defaults
    elif args.ablation == 'mrloss':
        parser.set_defaults(loss='margin_rank')
    args = parser.parse_args()

    if args.kg_model !='None':
        merged_relations = get_merged_relations(args.kg_name)
        args.relation_types = len(merged_relations)

        find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.dev_concepts, args.dev_rel_paths, args.nprocs, args.use_cache, merged_relations )
        find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.train_concepts, args.train_rel_paths, args.nprocs, args.use_cache, merged_relations )
        if args.test_statements is not None:
            find_relational_paths(args.cpnet_vocab_path, args.cpnet_graph_path, args.test_concepts, args.test_rel_paths, args.nprocs, args.use_cache, merged_relations )
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        eval(args)
    elif args.mode == 'pred':
        pred(args)
    else:
        raise ValueError('Invalid mode')


def train(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    config_path = os.path.join(args.save_dir, 'config.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    log_path = os.path.join(args.save_dir, 'log.csv')
    export_config(args, config_path)
    check_path(model_path)
    with open(log_path, 'w') as fout:
        fout.write('step,train_acc,dev_acc\n')

    ###################################################################################################
    #   Load data                                                                                     #
    ###################################################################################################

    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized, cp_emb = True, None
    else:
        use_contextualized = False
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1))

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)

    rel_emb = np.load(args.rel_emb_path)
    rel_emb = np.concatenate((rel_emb, -rel_emb), 0)
    rel_emb = cal_2hop_rel_emb(rel_emb)
    rel_emb = torch.tensor(rel_emb)
    relation_num, relation_dim = rel_emb.size(0), rel_emb.size(1)
    # print('| num_concepts: {} | num_relations: {} |'.format(concept_num, relation_num))

    device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() else 'cpu')

    # path_embedding_path = os.path.join('./path_embeddings/', args.dataset, 'path_embedding.pickle')
    if args.kg_model=='None':
        dataset = LMDataLoader(train_statement_path=args.train_statements, 
                               dev_statement_path=args.dev_statements,
                               test_statement_path=args.test_statements, 
                                batch_size=args.batch_size, 
                                eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, 
                                max_seq_length=args.max_seq_len, 
                                is_inhouse=args.inhouse, 
                                inhouse_train_qids_path=args.inhouse_train_qids,
                                subsample=args.subsample)
    elif args.kg_model!="gconattn":
        dataset = LMRelationNetDataLoader(args.path_embedding_path, args.train_statements, args.train_rel_paths,
                                      args.dev_statements, args.dev_rel_paths,
                                      args.test_statements, args.test_rel_paths,
                                      batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                                      model_name=args.encoder,
                                      max_tuple_num=args.max_tuple_num, max_seq_length=args.max_seq_len,
                                      is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                      use_contextualized=use_contextualized,
                                      train_adj_path=args.train_adj, dev_adj_path=args.dev_adj, test_adj_path=args.test_adj,
                                      train_node_features_path=args.train_node_features, dev_node_features_path=args.dev_node_features,
                                      test_node_features_path=args.test_node_features, node_feature_type=args.node_feature_type, 
                                      relation_types=args.relation_types, subsample=args.subsample)

    else:
        dataset = GconAttnDataLoader(train_statement_path=args.train_statements, train_concept_jsonl=args.train_concepts,
                                 dev_statement_path=args.dev_statements, dev_concept_jsonl=args.dev_concepts,
                                 test_statement_path=args.test_statements, test_concept_jsonl=args.test_concepts,
                                 concept2id_path=args.cpnet_vocab_path, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                 device=device, model_name=args.encoder, max_cpt_num=max_cpt_num[args.dataset],
                                 max_seq_length=args.max_seq_len, is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                 subsample=args.subsample, format=args.format, pretrained_concept_emb=cp_emb,text_only=args.text_only, concept2deg_path=None)
    ###################################################################################################
    #   Build model                                                                                   #
    ###################################################################################################

    lstm_config = get_lstm_config_from_args(args)
    if args.kg_model=='None':
        # model = LMForMultipleChoice(model_name=args.encoder, from_checkpoint=args.from_checkpoint, concept_num=concept_num, concept_dim=relation_dim, relation_num=relation_num, relation_dim=relation_dim, concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
        #                   hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num, num_attention_heads=args.att_head_num, fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
        #                   pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, freeze_ent_emb=args.freeze_ent_emb, init_range=args.init_range, ablation=args.ablation, use_contextualized=use_contextualized, emb_scale=args.emb_scale, encoder_config=lstm_config)
        model = LMForMultipleChoice(args.encoder, dropout=args.dropoutm, from_checkpoint=args.from_checkpoint, encoder_config=lstm_config)
    elif args.kg_model=='gconattn':
        # if args.ablation=='kg_only':
            # model = KGAttn(model_name=args.encoder, concept_num=concept_num,
                    #    concept_dim=relation_dim, concept_in_dim=concept_dim, freeze_ent_emb=args.freeze_ent_emb,
                    #    pretrained_concept_emb=cp_emb, hidden_dim=args.decoder_hidden_dim, dropout=args.dropoutm, encoder_config=lstm_config)
        # else:
        model = LMGconAttn(model_name=args.encoder, from_checkpoint=args.from_checkpoint, concept_num=concept_num,
                       concept_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim), concept_in_dim=concept_dim, freeze_ent_emb=args.freeze_ent_emb,
                       pretrained_concept_emb=cp_emb, hidden_dim=args.decoder_hidden_dim, dropout=args.dropoutm, ablation=args.ablation, encoder_config=lstm_config, lm_sent_pool=args.lm_sent_pool)
    else:
        model = LMRelationNet(model_name=args.encoder, from_checkpoint=args.from_checkpoint, 
                            concept_num=concept_num, concept_dim=relation_dim, relation_num=relation_num, relation_dim=relation_dim, concept_in_dim=(dataset.get_node_feature_dim() if use_contextualized else concept_dim),
                            hidden_size=args.mlp_dim, num_hidden_layers=args.mlp_layer_num, 
                            num_attention_heads=args.att_head_num,
                            fc_size=args.fc_dim, num_fc_layers=args.fc_layer_num, dropout=args.dropoutm,
                            pretrained_concept_emb=cp_emb, pretrained_relation_emb=rel_emb, 
                            freeze_ent_emb=args.freeze_ent_emb, init_range=args.init_range, 
                            ablation=args.ablation, use_contextualized=use_contextualized, emb_scale=args.emb_scale, encoder_config=lstm_config, kg_model=args.kg_model, lm_sent_pool=args.lm_sent_pool)

    try:
        model.to(device)
    except RuntimeError as e:
        print(e)
        print('best dev acc: 0.0 (at epoch 0)')
        print('final test acc: 0.0')
        print()
        return

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_parameters = [
        {'params': [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in model.decoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    if args.lr_schedule == 'fixed':
        scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps)

    print('parameters:')
    for name, param in model.decoder.named_parameters():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}'.format(name, param.size()))
        else:
            print('\t{:45}\tfixed\t{}'.format(name, param.size()))
    num_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print('\ttotal:', num_params)

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')

    ###################################################################################################
    #   Training                                                                                      #
    ###################################################################################################

    print()
    print('-' * 71)
    print(f'| batch_size: {args.batch_size} | num_epochs: {args.n_epochs} | num_train: {dataset.train_size()} |'
          f' num_dev: {dataset.dev_size()} | num_test: {dataset.test_size()}')
    global_step, best_dev_epoch = 0, 0
    best_dev_acc, final_test_acc, total_loss = 0.0, 0.0, 0.0
    start_time = time.time()
    model.train()
    if args.encoder_type=="PLM":
        freeze_net(model.encoder)
    # try:
    rel_grad = []
    linear_grad = []
    for epoch_id in tqdm(range(args.n_epochs), desc="Train Epoch"):
        if args.encoder_type=="PLM" and epoch_id == args.unfreeze_epoch:
            print('encoder unfreezed')
            unfreeze_net(model.encoder)
        if args.encoder_type=="PLM" and epoch_id == args.refreeze_epoch:
            print('encoder refreezed')
            freeze_net(model.encoder)
        model.train()
        for qids, labels, *input_data in dataset.train():
            optimizer.zero_grad()
            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                # print("labels:", labels[a:b])
                logits, _ = model(*[x[a:b] for x in input_data], layer_id=args.encoder_layer)
                # print("logits:", logits)
                # print(" ")

                if args.loss == 'margin_rank':
                    num_choice = logits.size(1)
                    flat_logits = logits.view(-1)
                    correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
                    correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
                    wrong_logits = flat_logits[correct_mask == 0]  # of length batch_size*(num_choice-1)
                    y = wrong_logits.new_ones((wrong_logits.size(0),))
                    loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
                elif args.loss == 'cross_entropy':
                    loss = loss_func(logits, labels[a:b])
                loss = loss * (b - a) / bs
                loss.backward()
                total_loss += loss.item()

            if (global_step + 1) % args.grad_step == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                optimizer.zero_grad()

            if (global_step + 1) % args.log_interval == 0:
                total_loss /= args.log_interval
                ms_per_batch = 1000 * (time.time() - start_time) / args.log_interval
                print('| step {:5} |  lr: {:9.7f} | loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss, ms_per_batch))
                # print('| rel_grad: {:1.2e} | linear_grad: {:1.2e} |'.format(sum(rel_grad) / len(rel_grad), sum(linear_grad) / len(linear_grad)))
                total_loss = 0
                rel_grad = []
                linear_grad = []
                start_time = time.time()
            global_step += 1

        model.eval()
        dev_acc = evaluate_accuracy(dataset.dev(), model)
        test_acc = evaluate_accuracy(dataset.test(), model) if dataset.test_size() > 0 else 0.0
        print('-' * 71)
        print('| epoch {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, dev_acc, test_acc))
        print('-' * 71)
        with open(log_path, 'a') as fout:
            fout.write('{},{},{}\n'.format(global_step, dev_acc, test_acc))
        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            final_test_acc = test_acc
            best_dev_epoch = epoch_id
            if args.save_model == 1:
                torch.save([model, args], model_path)
            print(f'model saved to {model_path}')
        model.train()
        start_time = time.time()
        if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break
    # except (KeyboardInterrupt, RuntimeError) as e:
    #     print(e)

    print()
    print('training ends in {} steps (best dev at epoch {})'.format(global_step, best_dev_epoch))
    print('best dev acc:\t{:.4f} '.format(best_dev_acc))
    print('final test acc:\t{:.4f}'.format(final_test_acc))
    print()


def eval(args):
    raise NotImplementedError()


def pred(args):

    dev_pred_path = os.path.join(args.save_dir, 'predictions_dev.json')
    test_pred_path = os.path.join(args.save_dir, 'predictions_test.json')
    model_path = os.path.join(args.save_dir, 'model.pt')
    device = torch.device('cuda:{}'.format(args.gpu_device) if torch.cuda.is_available() and args.cuda else "cpu")
    model, old_args = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized, cp_emb = True, None
    else:
        use_contextualized = False

    # path_embedding_path = os.path.join('./path_embeddings/', args.dataset, 'path_embedding.pickle')
    if args.kg_model=='None':
        dataset = LMDataLoader(train_statement_path=args.train_statements, 
                               dev_statement_path=args.dev_statements,
                               test_statement_path=args.test_statements, 
                                batch_size=args.batch_size, 
                                eval_batch_size=args.eval_batch_size,
                                device=device, model_name=args.encoder, 
                                max_seq_length=args.max_seq_len, 
                                is_inhouse=args.inhouse, 
                                inhouse_train_qids_path=args.inhouse_train_qids,
                                subsample=args.subsample)
    elif args.kg_model=='gconattn':
        dataset = GconAttnDataLoader(train_statement_path=args.train_statements, train_concept_jsonl=args.train_concepts,
                                 dev_statement_path=args.dev_statements, dev_concept_jsonl=args.dev_concepts,
                                 test_statement_path=args.test_statements, test_concept_jsonl=args.test_concepts,
                                 concept2id_path=args.cpnet_vocab_path, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
                                 device=device, model_name=args.encoder, max_cpt_num=max_cpt_num[args.dataset],
                                 max_seq_length=args.max_seq_len, is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                 subsample=args.subsample, format=args.format, pretrained_concept_emb=cp_emb, text_only=args.text_only, concept2deg_path=None)
    else:
        dataset = LMRelationNetDataLoader(args.path_embedding_path, old_args.train_statements, old_args.train_rel_paths,
                                      old_args.dev_statements, old_args.dev_rel_paths,
                                      old_args.test_statements, old_args.test_rel_paths,
                                      batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, device=device,
                                      model_name=old_args.encoder,
                                      max_tuple_num=old_args.max_tuple_num, max_seq_length=old_args.max_seq_len,
                                      is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
                                      use_contextualized=use_contextualized,
                                      train_adj_path=args.train_adj, dev_adj_path=args.dev_adj, test_adj_path=args.test_adj,
                                      train_node_features_path=args.train_node_features, dev_node_features_path=args.dev_node_features,
                                      test_node_features_path=args.test_node_features, node_feature_type=args.node_feature_type, relation_types=args.relation_types)
    print("***** generating model predictions *****")
    print(f'| dataset: {old_args.dataset} | save_dir: {args.save_dir} |')

    # for output_path, data_loader in ([(test_pred_path, dataset.test())] if dataset.test_size() > 0 else []):
    for data_loader, output_path in zip([dataset.dev(), dataset.test()], [dev_pred_path, test_pred_path] if dataset.test_size() > 0 else []):
        with torch.no_grad(), open(output_path, 'w') as fout:
            for qids, labels, *input_data in tqdm(data_loader):
                logits, _ = model(*input_data)
                for qid, label, logit in zip( qids, labels, logits):
                # for qid, pred_label in zip(qids, logits.argmax(1)):
                    pred_label = logit.argmax()
                    # fout.write('{},{}\n'.format(qid, chr(ord('A') + pred_label.item())))
                    # fout.write('{}\n'.format(pred_label))
                    out = {
                        "qid": qid,
                        "label": label.cpu().numpy().tolist(),
                        "pred_label": pred_label.cpu().numpy().tolist(),
                        "logits": [x.cpu().numpy().tolist() for x in logit ]
                    }
                    fout.write(json.dumps(out))
                    fout.write("\n")
                    # fout.write('{}{}\n'.format(pred_label))
        print(f'predictions saved to {output_path}')
    print('***** prediction done *****')

if __name__ == '__main__':
    main()
    sys.exit()