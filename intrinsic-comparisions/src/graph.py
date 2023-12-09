# modified based on  "chaitanya"
from collections import defaultdict, Counter
import nltk

class Graph:

    def __init__(self, directed=True):
        self.relations = defaultdict()
        self.nodes = defaultdict()
        self.node2id = {}
        self.relation2id = {}
        self.edges = {}
        self.edgeCount = 0
        self.directed = directed
        #self.add_node("UNK")
        #self.add_node("UNK-NODE")
        #self.add_relation("UNK-REL")

    def add_edge(self, node1, node2, rel, label, weight, uri=None, source=None):
        """

        :param node1: source node
        :param node2: target node
        :param rel: relation
        :param label: relation
        :param weight: weight of edge from [0.0, 1.0]
        :param uri: uri of edge
        :return: Edge object
        """
        new_edge = Edge(node1, node2, rel, label, weight, uri, source)

        if node2 in self.edges[node1]:
            self.edges[node1][node2].append(new_edge) # how to retrieve the edges index?
        else:
            self.edges[node1][node2] = [new_edge]

        # node1.neighbors.add(node2) #out-degree
        node2.neighbors.add(node1) #in-degree

        node1.degrees.add(node2) #our-degree
        node2.degrees.add(node1) #in-degree

        self.edgeCount += 1

        if (self.edgeCount + 1) % 10000 == 0:
            print("Number of edges: %d" % self.edgeCount, end="\r")

        return new_edge

    def load_node(self, name, idx):
        """

        :param name:
        :return:
        """
        new_node = Node(name, idx+1)
        self.nodes[idx+1] = new_node
        self.node2id[new_node.name] = idx
        self.edges[new_node] = {}
        return self.node2id[new_node.name]

    def add_node(self, name):
        """

        :param name:
        :return:
        """
        new_node = Node(name, len(self.nodes))
        self.nodes[len(self.nodes)] = new_node
        self.node2id[new_node.name] = len(self.nodes) - 1
        self.edges[new_node] = {}
        return self.node2id[new_node.name]

    def add_relation(self, name):
        """
        :param name
        :return:
        """
        new_relation = Relation(name, len(self.relations))
        self.relations[len(self.relations)] = new_relation
        self.relation2id[new_relation.name] = len(self.relations) - 1
        return self.relation2id[new_relation.name]

    def find_node(self, name):
        """
        :param name:
        :return:
        """
        if name in self.node2id:
            return self.node2id[name]
        else:
            return -1

    def find_relation(self, name):
        """
        :param name:
        :return:
        """
        if name in self.relation2id:
            return self.relation2id[name]
        else:
            return -1

    def is_connected(self, node1, node2):
        """

        :param node1:
        :param node2:
        :return:
        """
        if node1 in self.edges:
            if node2 in self.edges[node1]:
                return True
        return False

    def node_exists(self, node):
        """

        :param node: node to check
        :return: Boolean value
        """
        if node in self.nodes.values():
            return True
        return False

    def find_all_connections(self, relation):
        """
        :param relation:
        :return: list of all edges representing this relation
        """
        relevant_edges = []
        for edge in self.iter_edges():
            if edge.relation.name == relation:
                relevant_edges.append(edge)
        return relevant_edges

    def find_relation_distribution(self):
        r_ht = {}
        for relation, id in self.relation2id.items():
            relation_edges = self.find_all_connections(relation)
            r_ht[relation]= len(relation_edges)
        r_ht_sorted = sorted(r_ht.items(), key= lambda item: item[1], reverse=True)
        return r_ht_sorted

    def node_tokens_len_statistics(self):
        len2cnt=Counter()
        nodes = self.iter_nodes()
        for node in nodes:
            word = node.name.replace("_", " ")
            tokens = nltk.word_tokenize(word)
            length = len(tokens)
            if length ==1:
                print(tokens)
            len2cnt[length] = len2cnt[length]+1

        words_per_concept    = sum([k*v for k,v in len2cnt.items()])/ sum(len2cnt.values())
        mutli_word_ratio = sum([v for k,v in len2cnt.items()  if k>1 ])/ sum(len2cnt.values()) 

        return words_per_concept, mutli_word_ratio, len2cnt

    def iter_nodes(self):
        return list(self.nodes.values())

    def iter_relations(self):
        return list(self.relations.values())

    def iter_edges(self):
        for node in self.edges:
            for edge_list in self.edges[node].values():
                for edge in edge_list:
                    yield edge

    def __str__(self):
        for node in self.nodes:
            print(node)

    def find_rel_by_node_name(self, node1_name, node2_name, weight=False):
        '''if relation exists, return the relation list;
            else: return None
        '''
        rel = None
        if node1_name in self.node2id and node2_name in self.node2id:
            node1_id = self.node2id[node1_name]
            node2_id = self.node2id[node2_name]

            node1 = self.nodes[node1_id]
            node2 = self.nodes[node2_id]

            flag = self.is_connected(node1, node2)
            if flag:
                edge = self.edges[node1][node2]
                if not weight:
                    rel = [ e.relation.name for e in edge]
                else:
                    rel = [ (e.relation.name, e.weight) for e in edge]
        return rel

    def find_neighborhoods(self, node_name):
        if node_name in self.node2id:
            neighbors = list()
            for i, edge in enumerate(self.iter_edges()):
                if edge.src.name == node_name or edge.tgt.name == node_name:
                    neighbors.append([edge.src.name, edge.tgt.name, edge.relation.name, str(edge.weight)])
            return neighbors
        else:
            print("Not found {}".format(node_name))
            return None


class Node:

    def __init__(self, name, id, lang='en'):
        self.name = name
        self.id = id
        self.lang = lang
        self.neighbors = set([])
        self.degrees = set([])

    def get_neighbors(self):
        """
        :param node:
        :return:
        """
        return self.neighbors

    def get_degree(self):
        """

        :param node:
        :return:
        """
        # return len(self.neighbors)
        return len(self.degrees)

    def __str__(self):
        out = ("Node #%d : %s" % (self.id, self.name))
        return out


class Relation:

    def __init__(self, name, id):
        self.name = name
        self.id = id


class Edge:

    def __init__(self, node1, node2, relation, label, weight, uri, source):
        self.src = node1
        self.tgt = node2
        self.relation = relation
        self.label = label
        self.weight = weight
        self.uri = uri
        self.source = source

    def __str__(self):
        if self.relation is not None:
            out = "{}: {} --> {} \t\t {}: {} --> {}".format(self.relation.name, self.src.name, self.tgt.name, self.relation.id, self.src.id, self.tgt.id)
        else:

            out = ("%s: %s --> %s" % (self.weight, self.src.name, self.tgt.name))
        return out
