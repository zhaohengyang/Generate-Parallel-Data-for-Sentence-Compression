
import copy
import pydot
import os
from matplotlib.pyplot import imshow
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


import spacy
from spacy.en import English
from spacy.tokens.span import Span
nlp = English()
import copy
from nltk.stem.porter import PorterStemmer

from spacy.attrs import ORTH, DEP, HEAD


merge_rules = {"group1": ["aux", "auxpass", "det", "nummod", "case",
                          "prt", "poss", "of", "nmod", "compound",
                          "neg", "xcomp", "quantmod", "advmod", "attr",
                          "pobj", "as", "aux", "dobj", "amod",
                          "npadvmod"],
               "group2": ["cc"],
               "group3": ["mark", "," ]}

worked = {"npadvmod": [30, 37], "to": [40], "mark": [67], "attr": [58], "pobj": [69], "punct": [69],
          "conj": [69], "dobj": [77], "nsubj": [84, 86], "amod": [98], "ccomp": [4]}

not_worked = {"npadvmod": [30], "mark": [41], "cc": [38], "advcl": [41], "pobj": [45], "with": [63],
              "conj": [71, 76], "nsubj": [47, 77], "ccomp": [16]}


# Resize and clean edges
def plot_im(im, dpi=80):
    py,px,_ = im.shape # depending of your matplotlib.rc you may have to use py,px instead
    size = (py/np.float(dpi), px/np.float(dpi)) # note the np.float()

    fig = plt.figure(figsize=size, dpi=dpi)
    # fig = plt.figure(figsize=(10,20), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # Customize the axis
    # remove top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    # turn off ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.imshow(im)
    plt.show()


def get_decode(s):
    return unicode(s).encode("utf-8")


class Tree_node():

    # Initialize tree
    def __init__(self, node):
        self.node = node

    # Get node's id
    def id(self):
        return self.node[u'word'][self.node['head_word_index']]['id']

    # Get head word tag
    def head_word_tag(self):
        return self.node[u'word'][self.node['head_word_index']]['tag']

    # Get head word stem
    def head_word_stem(self):
        return self.node[u'word'][self.node['head_word_index']]['stem']

    # Get head word
    def head_word(self):
        return self.node[u'word'][self.node['head_word_index']]

    # Get tag of each word in node
    def tags(self):
        return [word['tag'] for word in self.node['word']]

    # Get stem of each word in node
    def stems(self):
        return [word['stem'] for word in self.node['word']]

    # Get form of each word in node
    def forms(self):
        return [word['form'] for word in self.node['word']]

    # Get id of each word in node
    def ids(self):
        return [word['id'] for word in self.node['word']]


    # Get edge
    def edge(self):
        return self.node['edge']

    # Get form
    def form(self):
        return self.node['form']

    # Get edge label
    def edge_label(self):
        return self.node['edge']['label']

    # Get edge parent id
    def edge_parent_id(self):
        return self.node['edge']['parent_id']

    # Get word
    def word(self):
        return self.node['word']

    # Set new parent id
    def set_parent_id(self, parent_id):
        self.node['edge']['parent_id'] = parent_id

    # Set new form
    def set_form(self, form):
        self.node['form'] = form

    # Set new word
    def set_word(self, word):
        self.node['word'] = word

    # Set head word index
    def set_head_word_index(self, index):
        self.node['head_word_index'] = index

    # Set edge label
    def set_edge_label(self, label):
        self.node['edge']['label'] = label

    # Show form id combination
    def node_forms_and_ids(self):
        return " ".join([word[u'form'] + "_" + str(word[u'id'])
                         for word in self.word()])

    # Show node
    def describe(self):
        return get_decode("node: {:<20} head_word_id:{:<20}".format(self.node_forms_and_ids(), self.id()))



class Parsed_Tree():
    '''
    Manage and maintain an Tree
    '''

    # Initialize from string
    def __init__(self, nodes):
        self.tree = nodes

    def get_copy(self):
        return copy.deepcopy(self)

    # Delete node
    def remove_node(self, node):
        self.tree.remove(node)

    # Add node
    def append_node(self, node):
        self.tree.append(node)

    # Any children of A will point to B
    def update_children(self, A, B):
        for child in self.children(A):
            child.set_parent_id(B.id())

    # Merge A to B (parent of A)
    def merge(self, A, B):
        parent_head_word = B.head_word()['form']

        new_word = A.word() + B.word()
        new_word.sort(key=lambda x: x['id'])
        word_list = [word[u"form"] for word in new_word]

        B.set_word(new_word)
        B.set_form(" ".join(word_list))
        B.set_head_word_index(word_list.index(parent_head_word))

        self.update_children(A, B)
        self.remove_node(A)

    # Insert between A, B(child of A)
    def insert_between(self, node, A, B):
        # node point to A
        node.set_parent_id(A.id())
        # B point to node
        B.set_parent_id(node.id())

    # Get children node
    def children(self, node):
        children = []
        for child_node in self.tree:
            if node.id() == child_node.edge_parent_id():
                children.append((child_node))
        return children

    # Find parent node
    def find_parent_node(self, node):
        return self.find_node_by_id(node.edge_parent_id())

    # check consistency
    def consistency(self):
        assert all([self.find_parent_node(node) for node in self.tree])
        assert len(self.all_roots()) == 1

    # check if node is root node
    def is_root(self, node):
        return node.id() == node.edge_parent_id()

    # Get all root nodes if it has more than one (shouldn't), used for check consistency
    def all_roots(self):
        # The nodes is a tree, each node has one edge to it's parent
        return [node for node in self.tree if self.is_root(node)]

    # Get root node
    def root_node(self):
        return self.all_roots()[0]

    # Get path to root
    def path_to_root(self, node):
        path = []
        current_node = node
        while not self.is_root(current_node):
            path.append(current_node)
            current_node = self.find_parent_node(current_node)
        path.append(current_node)
        return path

    # Get path from A to B
    def path(self, A, B, debug = False):
        self.consistency()
        A_path_root = [node.id() for node in self.path_to_root(A)]
        B_path_root = [node.id() for node in self.path_to_root(B)]
        B_path_root.reverse()

        joined = set(A_path_root) & set(B_path_root)

        up = copy.deepcopy(A_path_root)

        [up.remove(item) for item in joined]

        down = copy.deepcopy(B_path_root)

        [down.remove(item) for item in joined]

        [A_path_root.remove(item) for item in up]

        top = [] + A_path_root[:1]

        if debug:
            print("up:", up)
            print("top", top)
            print("down:", down)

        return up, top, down

    # Add an dummy on top of original root
    def add_dummy_root(self):
        # -- Add dummy root node
        # Create an dummy root node, append it to node list
        dummy_root_id = -1
        dummy_root = {u'form': u'ROOT',
                      u'head_word_index': 0,
                      u'word': [{u'tag': u'ROOT',
                                 # u'dep': u'ROOT_To_Self',
                                 u'id': dummy_root_id,
                                 u'form': u'ROOT',
                                 u'stem': u'ROOT'}],
                      u'edge': {u'parent_id': dummy_root_id, u'label': u'ROOT_To_Self'}
                      }

        self.append_node(Tree_node(dummy_root))

        # Find original root node, which contains self pointed edge
        root_node = self.root_node()
        root_node.set_parent_id(dummy_root_id)

    # Get node given id
    def find_node_by_id(self, id, debug=False):
        found = None
        if debug:
            print("Debug ----- find_node_by_id ----- ")
            print("target id:", id)
            print([node.id() for node in self.tree])
        for node in self.tree:
            if id in node.ids():
                found = node
                break
        return found

    # Check if node is in the tree
    def is_node_in(self, node):
        if self.find_node_by_id(node.id()):
            return True
        else:
            return False

    # Find neighbor nodes
    def find_neighbor(self, node, debug = False):
        node_ids = [tree_node.id() for tree_node in self.tree]
        node_ids.sort()

        rights = filter(lambda x: x > node.id(), node_ids)[:1]
        right = next(iter(rights), None)

        node_ids.reverse()
        lefts = filter(lambda x: x < node.id(),node_ids )[:1]
        left = next(iter(lefts), None)

        left_node = self.find_node_by_id(left) if left else None
        right_node = self.find_node_by_id(right) if right else None
        if debug:
            print("find_neighbor:")
            print("ids: {}".format(node_ids))
            print("node: {}, left: {}, right: {}".format(node.id(), left, right))
        return left_node, right_node

    # Print tree
    def print_edges(self, debug=False):
        def get_tags(node):
            return get_decode(",".join(node.tags()))

        for node in self.tree:
            parent_id = node.edge_parent_id()
            parent_node = self.find_node_by_id(parent_id)
            if node.edge_label():
                if not parent_node:
                    self.find_node_by_id(parent_id, debug=True)
                    print(get_decode("{:<20}:{:<20}{}->{}".format(node.id(), node.edge_label(), node.form(), parent_id)))
                else:
                    print(get_decode("{:<20}:{:<20}{}->{}[{}]".format(node.id(), node.edge_label(),
                                                               node.form(),
                                                               parent_node.form(),
                                                               get_tags(parent_node)
                                                               )))
    # Print a graphic tree
    def print_graph(self, color_settings = {}):
        graph = pydot.Dot(graph_type='digraph')
        name_to_node = {}

        id_to_color = {}
        for color, ids in color_settings.items():
            for id in ids:
                id_to_color[id] = color


        all_ids = [node.id() for node in self.tree]
        for id in all_ids:
            color = id_to_color.get(id, 'gray')
            node = self.find_node_by_id(id)
            node_label = node.node_forms_and_ids()
            name_to_node[node_label] = pydot.Node(node_label, style="filled", fillcolor=color)


        for node in name_to_node.values():
            graph.add_node(node)

        for edge in self.get_edges():
            node_a_name, node_b_name = edge
            graph.add_edge(pydot.Edge(name_to_node[node_a_name], name_to_node[node_b_name]))

        graph.write_png('graph.png')
        img = mpimg.imread('graph.png')

        plot_im(img, dpi=40)
        os.remove("graph.png")

    # Get and edge from a node to its parent
    def get_edge(self, node):
        parent_id = node.edge_parent_id()
        parent_node = self.find_node_by_id(parent_id)
        return (parent_node.node_forms_and_ids(), node.node_forms_and_ids())

    # Get tree edges
    def get_edges(self):
        return [self.get_edge(node) for node in self.tree]







class Sentence_Reduction(object):
    def __init__(self, sentence_tree, headline_tree):
        # parse sentence into tree structure
        self.sentence_tree = sentence_tree
        # parse headline into tree structure
        self.headline_tree = headline_tree
        # Transfer headline into transfered_headline
        self.transfered_headline = None
        # Transfer sentence_tree into transfered_tree
        self.transfered_tree = None
        # Flat transfered_tree into flatten_tree
        self.flatten_tree = None
        # Given headline_tree, reduce flatten_tree into reduced_tree
        self.reduced_tree = None
        # Get reduced_sentence from reduced_tree
        self.reduced_sentence = None

    def transfer_tree(self, debug=False):
        # Start
        self.transfered_tree = self.sentence_tree.get_copy()
        self.transfered_tree.consistency()

        # Add dummy root
        self.transfered_tree.add_dummy_root()

        # Remove node that falls in ignore rules
        def ignore_node(node):
            # ignore_rules = ["``", "''", "'"]
            ignore_rules = []
            if node.form() in ignore_rules:
                parent_node = self.transfered_tree.find_parent_node(node)
                self.transfered_tree.update_children(node, parent_node)
                return True
            else:
                return False

        # remove node in ignore rules
        self.transfered_tree.tree[:] = [node for node in self.transfered_tree.tree if not ignore_node(node)]

        # -- preposition, punctuation replacement
        part_of_speach = ['prep', 'punct']
        for node in self.transfered_tree.tree:
            if node.edge_label() in part_of_speach:
                node.set_edge_label(node.head_word()['form'])

        # -- move conjunction word
        for node in self.transfered_tree.tree:
            if node.edge_label() in merge_rules['group2']:
                # print("Found cc node: node label: {:<20}  id: {:<20} form: {:<20}".format(node.edge_label(), node.id(), node.form()))
                _, right_neighbor = self.transfered_tree.find_neighbor(node)
                up, top, down = self.transfered_tree.path(node, right_neighbor)
                if up and down:
                    A_node = self.transfered_tree.find_node_by_id(top[0])
                    B_node = self.transfered_tree.find_node_by_id(down[0])
                    self.transfered_tree.insert_between(node, A_node, B_node)

    # Take a transfered tree and flat it
    def flat_tree(self):
        self.flatten_tree = self.transfered_tree.get_copy()
        self.flatten_tree.consistency()

        # Remove node each time after merged to its parent node
        for node in list(self.flatten_tree.tree):
            if node.edge_label() in merge_rules['group1']:
                self.flatten_tree.merge(node, self.flatten_tree.find_parent_node(node))


    # Transfer headline
    def transfer_headline(self, debug = False):
        self.transfered_headline = self.headline_tree.get_copy()
        self.transfered_headline.consistency()

        # Remove node that falls in ignore rules
        def ignore_node(node):
            headline_ignore_rules = ['IN', '``', "''", "DT", ':', '.', 'POS']
            if node.head_word_tag() in headline_ignore_rules:
                parent_node = self.transfered_headline.find_parent_node(node)
                self.transfered_headline.update_children(node, parent_node)
                return True
            else:
                return False

        if debug:
            for node in self.headline_tree.tree:
                print("{}--{}".format(get_decode(node.head_word_stem()), get_decode(node.head_word_tag())))
        self.transfered_headline.tree[:] = [node for node in self.transfered_headline.tree if not ignore_node(node)]


    def reduce_sentence_by_headline(self, addtional_stem = None, debug=False):
        # def is_verb(tag):
        #     return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

        def stemming(node):
            if addtional_stem:
                stems = [addtional_stem(form.lower()) for form in node.forms()]
            else:
                stems = [stem.lower() for stem in node.stems()]
            return " ".join(stems)

        def get_node_set(node):
            stem = stemming(node)
            return stem.split()

        def is_not_in_headline(node, headline_stems):
            return not bool(set(get_node_set(node)) & set(headline_stems))

        def check_common_and_update(node, debug = False):
            node_stems = get_node_set(node)
            common = set(node_stems) & set(headline_stems)

            if bool(common):
                for item in common:
                    headline_stems.remove(item)
                if debug:
                    print("modified_headline_stems: ", headline_stems)

                return True
            else:
                return False

        # Start
        self.reduced_tree = self.flatten_tree.get_copy()
        self.reduced_tree.consistency()

        # Get a list of stems and flatten the list
        headline_stems = [get_node_set(headline_node) for headline_node in self.transfered_headline.tree]
        headline_stems = [item for itemset in headline_stems for item in itemset]

        # Get a list of connect words for later use
        connect_nodes = [node for node in self.reduced_tree.tree if node.edge_label() in merge_rules['group3']]

        # Keep node that has headline stem
        self.reduced_tree.tree[:] = [node for node in self.reduced_tree.tree if check_common_and_update(node)]

        if headline_stems:
            print("{} -- Found unmatched headlines: {}".format(self.reduce_sentence_by_headline.__name__, headline_stems))


        # Return each part of the flatten tree, use different color to print graph
        reduced_tree_ids = [n.id() for n in self.reduced_tree.tree]

        # Add node on the path to reduced tree
        nodes_on_the_path = []
        processed = []
        for index, node in enumerate(self.reduced_tree.tree):
            path = self.flatten_tree.path_to_root(node)
            path_label = [self.reduced_tree.is_node_in(node) for node in path]
            # the last item is zero(dummy root)
            # Find the first True and the last True
            # Index in between will be added to reduced graph
            start, end = path_label.index(True) + 1, len(path_label) - path_label[::-1].index(True) - 1

            for node_on_path in path[start:end]:
                if not self.reduced_tree.is_node_in(node_on_path) and node_on_path.id() not in processed:
                    # print("add current node: {}".format(path[node_index]['form']))
                    processed.append(node_on_path.id())
                    nodes_on_the_path.append(node_on_path)

        # Return each part of the flatten tree, use different color to print graph
        path_node_ids = [n.id() for n in nodes_on_the_path]

        self.reduced_tree.tree += nodes_on_the_path

        def use_connect_word(node):
            # Find connnect word like "that" or "which", we use them only if both left words
            # and right words are selected in reduced tree.
            # The left word is the word right before the connect word
            # The right word is any word after connect
            left_word_id = node.id() - 1
            left_node, right_node = self.reduced_tree.find_neighbor(node)
            return self.reduced_tree.find_node_by_id(left_word_id) and right_node

        # Add connect word fot reduced tree if needed
        connect_nodes[:] = [node for node in connect_nodes if use_connect_word(node) and not self.reduced_tree.is_node_in(node)]

        for node in connect_nodes:
            print("Found connect node: {}".format(node.describe()))

        # Return each part of the flatten tree, use different color to print graph
        connect_nodes_ids = [n.id() for n in connect_nodes]

        self.reduced_tree.tree += connect_nodes

        # Make reduced tree consistent
        for reduced_node in self.reduced_tree.tree:
            if not self.reduced_tree.find_parent_node(reduced_node):
                reduced_node.set_parent_id(reduced_node.id())


        return reduced_tree_ids, path_node_ids, connect_nodes_ids

    # Generate reduced sentence from reduced node
    def generate_reduced_sentence(self):
        id_word_pairs = [(word[u'id'], word[u'form']) for reduced_node in self.reduced_tree.tree
                         for word in reduced_node.word()]

        id_word_pairs.sort(key=lambda tuple: tuple[0])

        self.reduced_sentence = " ".join([tuple[1] for tuple in id_word_pairs])




# Construct tree from sentence
def parse_info(sentence):
    doc = nlp(sentence)

    heads = [index + item[0] for index, item in enumerate(doc.to_array([HEAD]))]

    nodes = [{u"form": token.orth_,
              u"head_word_index": 0,
              u"word": [{u"id": current_id,
                         # u"dep": doc[current_id].dep_,
                         u"tag": token.tag_,
                         u"form": token.orth_,
                         u"stem": token.lemma_
                         }],
              u"edge": {u"parent_id": parent_id, u"label": doc[current_id].dep_}
              }
             for current_id, (token, parent_id) in enumerate(zip(doc, heads))]

    return [Tree_node(node) for node in nodes]