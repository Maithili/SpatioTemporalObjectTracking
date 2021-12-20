import networkx as nx
from math import sqrt, floor, ceil
from networkx.algorithms.components import connected
from networkx.classes import graph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from torch.nn import functional as F

from encoders import time_external

EDGE_THRESH = 0.1
SEED = 456
SINGLE_PLOT = False

category_colors = {
    "Rooms": "#5E4955",
    "Furniture": "#996888",
    "Appliances": "#3F273B",
    "placable_objects": "#C99DA3"
}

node_colors_by_active = ["#C99DA3", "#996888"]

days = ["Monday", 'Tuesday', "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def _get_node_info(graph_nodes, node_classes):
    node_names = []
    for node in list(graph_nodes):
        node_names.append(node_classes[node.argmax()])
    return node_names

def _draw_graph(G, ax=None, pos=None, colors = "#996888"):
    if pos is None:
        pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw_networkx(G, pos=pos, ax=ax, edge_cmap=plt.cm.Blues, edge_vmin = 0 ,edge_vmax = 1 , edgelist=G.edges(), edge_color=weights, node_size=1000, node_color=colors)
    return pos


def _visualize_graph(graph_nodes, graph_edges, node_classes, ax=None, pos=None, node_categories=[], node_color=None):
    G = nx.DiGraph()
    graph_node_names = _get_node_info(graph_nodes, node_classes)
    colors = "#996888"
    if node_color is not None:
        colors = node_color
    elif node_categories:
        categories = _get_node_info(graph_nodes, node_categories)
        colors = [category_colors[c] for c in categories]
    G.add_nodes_from(graph_node_names)
    
    nodes_from, nodes_to = np.argwhere(graph_edges > EDGE_THRESH)

    for n_from, n_to in zip(list(nodes_from), list(nodes_to)):
        G.add_edge(graph_node_names[n_from], graph_node_names[n_to], weight=float(graph_edges[n_from][n_to]))
    
    return _draw_graph(G, ax=ax, pos=pos, colors=colors)


def visualize_datapoint(model, dataloader, node_classes, node_categories = [], use_output_nodes = False):
    data_list = list(dataloader)
    inp = input('Do you want to visualize any output? (y/n)')

    while(inp == 'y' and len(data_list)>0):
        data = data_list.pop()
        eval, details = model.step(data)
        
        colors = [node_colors_by_active[act] for act in details['evaluate_node'].squeeze(0)]

        fig, axs = plt.subplots(2,2)

        ax = axs[0][0]
        positions = _visualize_graph(F.one_hot(details['input']['class'].squeeze(0)), F.one_hot(details['input']['location'].squeeze(0)), node_classes, ax = ax, node_categories=node_categories, node_color=colors)
        ax.set_title('Input')

        ax = axs[1][0]
        if use_output_nodes:
            _visualize_graph(F.one_hot(details['output']['class'].squeeze(0)), F.softmax(details['output_probs']['location'].squeeze(0).squeeze(-1), dim=-1), node_classes, ax = ax, pos=positions, node_categories=node_categories, node_color=colors)
        else:
            _visualize_graph(F.one_hot(details['input']['class'].squeeze(0)), F.softmax(details['output_probs']['location'].squeeze(0).squeeze(-1), dim=-1), node_classes, ax = ax, pos=positions, node_categories=node_categories, node_color=colors)
        ax.set_title('Predicted (probabilities)')
            
        ax = axs[1][1]
        if use_output_nodes:
            _visualize_graph(F.one_hot(details['output']['class'].squeeze(0)), F.one_hot(details['output']['location'].squeeze(0)), node_classes, ax = ax, pos=positions, node_categories=node_categories, node_color=colors)
        else:
            _visualize_graph(F.one_hot(details['input']['class'].squeeze(0)), F.one_hot(details['output']['location'].squeeze(0)), node_classes, ax = ax, pos=positions, node_categories=node_categories, node_color=colors)
        ax.set_title('Predicted')
        
        ax = axs[0][1]
        _visualize_graph(F.one_hot(details['gt']['class'].squeeze(0)), F.one_hot(details['gt']['location'].squeeze(0)), node_classes, ax = ax, pos=positions, node_categories=node_categories, node_color=colors)
        ax.set_title('Expected')

        fig.suptitle('Loss : '+str(eval['losses']['mean'])+' ; Accuracy : '+str(eval['accuracy']))
        
        plt.show()
        plt.savefig('temp.jpg')
        inp = input('Do you want to visualize another output? (y/n)')



def visualize_routine(routine, dt=10, sparsify=False):
    graphs = routine['graphs']
    times = routine['times']
    num_plots = len(graphs)
    num_x = int(floor(sqrt(num_plots)))
    num_y = int(ceil(num_plots/num_x))
    fig, axs = plt.subplots(num_x,num_y)
    axs = axs.reshape(-1,)
    pos=None
    for i,(graph,t) in enumerate(zip(graphs,times)):
        node_labels = {n['id']:n['class_name'] for n in graph['nodes']}
        node_ids = [n['id'] for n in graph['nodes']]
        node_colors = [category_colors[n['category']] for n in graph['nodes']]
        G = nx.DiGraph()
        G.add_nodes_from(node_ids)
        for edge in graph['edges']:
            if edge['relation_type'] in ["INSIDE","ON"]:
                n1, n2 = edge['from_id'], edge['to_id']
                G.add_edge(n1, n2, weight=1)
        
        if sparsify:
            for nid in node_ids:
                direct_edges = [to_1 for fro_1,to_1 in G.edges() if fro_1 == nid]
                connected_HO = set([to_2 for fro_2,to_2 in G.edges() for to_1 in direct_edges if fro_2 == to_1])
                for _ in range(10):
                    next_connection = [to_2 for fro_2,to_2 in G.edges() for to_1 in connected_HO if fro_2 == to_1]
                    if not next_connection:
                        break
                    connected_HO.update(next_connection)
                for to in direct_edges:
                    if to in connected_HO:
                        G.remove_edge(nid,to)

        if pos is None:
            layers = category_colors.values()
            nlist = [[i for i,c in zip(node_ids,node_colors) if c==color] for color in layers]
            pos = nx.shell_layout(G, nlist=nlist)
            # pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, ax=axs[i], edge_cmap=plt.cm.Blues, edge_vmin = 0 ,edge_vmax = 1, node_size=300, node_color=node_colors, labels = node_labels)
        th = time_external(t*dt)
        axs[i].set_title(str(int(th[2]))+':'+str(int(th[3])))
    th = [int(t) for t in th]
    fig.suptitle("Week "+str(th[0])+" - "+days[th[1]])
    plt.show()
    plt.savefig('temp.jpg')
    
