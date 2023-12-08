import csv

def write_eval_to_files(net, outpath, suffix, input_order):
    print("Writing {} files".format(suffix))
    sampled_nodes = set(net.graph.node2id.keys())
    sampled_rels  = set(net.graph.relation2id.keys())
    sampled_edges = net.edge_set

    print("# Sampled overlap edges: {}, nodes: {}, rels: {}".format(
            len(sampled_edges), len(sampled_nodes), len(sampled_rels)))

    write_relation_triples(sampled_edges, outpath+'/{}'.format(suffix), inp_order=input_order)
    write_links(sampled_nodes, outpath=outpath+'/ent_links_{}'.format(suffix), link_type='nodes')
    write_links(sampled_rels, outpath=outpath + '/rel_links_{}'.format(suffix), link_type='relation')
    print()


def write_train_to_files(outpath, net_cn, net_sw, overlap_edges, overlap_nodes, overlap_rels):
    print("Writing training files")
    write_relation_triples(net_cn.edge_set, outpath +'/rel_triples_1', 'rht')
    write_relation_triples(net_sw.edge_set, outpath +'/rel_triples_2', 'rht')

    write_links(overlap_nodes, outpath + '/ent_links_{}'.format('train'), link_type='nodes')
    write_links(overlap_rels, outpath=outpath + '/rel_links', link_type='relation')

def write_entity(entities, outpath):
    with open(outpath, "w") as f:
        if isinstance(entities, set):
            for e in entities:
                f.write("{}\n".format(e))
    print("# Write {} entities to {}".format(len(entities),outpath))

def write_links( links, outpath, link_type='relation'):
    '''
    links: a dict, links[rel/ent]= freq
    return: list(rel: rel)
    '''
    writer = open(outpath,"w")
    cnt=0
    if isinstance(links, dict):
        for link, freq  in links.items():
            out_line="{}\t{}\n".format(link, link)
            writer.write(out_line)
            cnt+=1

    if isinstance(links, set):
        for link in links:
            out_line="{}\t{}\n".format(link, link)
            writer.write(out_line)
            cnt+=1

    print("# Write {} overlap {} to {}".format(cnt, link_type, outpath))
    if link_type=='relation' and isinstance(links, dict):
        rel_unlinks=set()
        for k,v in net_cn.relation_count:
            if k not in links.keys():
                rel_unlinks.add(k)
        links= sorted(links.items(), key=lambda x: x[1])

def write_relation_triples( triples, outpath, inp_order="rht"):
    '''
    triples: dict, (arg1, arg2, rel)
    return: write (arg1, rel, arg2), out_order=(rht)
    '''
    writer = open(outpath,"w")
    cnt=0

    if isinstance(triples, list):
        if inp_order=='hrt':
            for arg1, rel, arg2 in triples:
                #if rel in ["AtLocation", "IsA"]:
                out_line="{}\t{}\t{}\n".format(arg1, rel, arg2)
                writer.write(out_line)
                cnt+=1

    if inp_order=='htr':
        for arg1, rels in triples.items(): 
            for arg2, rel in rels.items(): 
                #if rel in ["AtLocation", "IsA"]:
                out_line="{}\t{}\t{}\n".format(arg1, rel, arg2)
                writer.write(out_line)
                cnt+=1
    
    if inp_order=='rht':
        if isinstance(triples, dict):
            for rel, args in triples.items(): 
                for arg1, arg2 in args.items(): 
                    #if rel in ["AtLocation", "IsA"]:
                    out_line="{}\t{}\t{}\n".format(arg1, rel, arg2)
                    writer.write(out_line)
                    cnt+=1

        if isinstance(triples, list) or isinstance(triples, set):
            for rel, arg1, arg2 in triples: 
                    out_line="{}\t{}\t{}\n".format(arg1, rel, arg2)
                    writer.write(out_line)
                    cnt+=1
  

    if inp_order=='rhtw':
        if isinstance(triples, list) or isinstance(triples, set):
            for rel, arg1, arg2, weight in triples: 
                writer.write('\t'.join([rel, arg1, arg2, str(weight)]) + '\n')
                cnt+=1
 
    print("# Write %d triples to %s"%(cnt , outpath))
    writer.close()
    

def write_nodes(nodes, outpath):
    with open(outpath, "w") as fout:
        for node in nodes: 
            fout.write(node + '\n')
    print("# Write %d lines to %s"%(len(nodes), outpath))

def write_node2degree(node2degree, outpath):
    with open(outpath, "w") as fout:
        for (node, degree) in node2degree: 
            fout.write(node + '\t' + str(degree) + '\n')
    print("# Write %d  lines to %s"%(len(node2degree), outpath))