import networkx as nx
from math import sqrt, floor, ceil
from networkx.algorithms.components import connected
from networkx.classes import graph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

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

days = ["Monday", 'Tuesday', "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def _get_node_info(graph_nodes, node_classes):
    node_names = []
    for node in list(graph_nodes):
        node_names.append(node_classes[node.argmax()])
    return node_names

def _draw_graph(G, graphs_per_edge_class, edge_classes, axs=None, pos=None, colors = "#996888"):
    if axs is None:
        _, axs = plt.subplots(1,len(edge_classes))
    if pos is None:
        pos = nx.spring_layout(G)
    for ax,edge_cl in zip(axs, edge_classes):
        if edge_cl == "CLOSE" and SINGLE_PLOT:
            continue
        weights = [graphs_per_edge_class[edge_cl][u][v]['weight'] for u,v in graphs_per_edge_class[edge_cl].edges()]
        nx.draw_networkx(G, pos=pos, ax=ax, edge_cmap=plt.cm.Blues, edge_vmin = 0 ,edge_vmax = 1 , edgelist=graphs_per_edge_class[edge_cl].edges(), edge_color=weights, node_size=1000, node_color=colors)
        nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels={k:edge_cl for k in graphs_per_edge_class[edge_cl].edges()})
        ax.set_title(edge_cl)
    return pos


def _visualize_graph(graph_nodes, graph_edges, node_classes, edge_classes, axs=None, pos=None, node_categories=[]):
    G = nx.DiGraph()
    graph_node_names = _get_node_info(graph_nodes, node_classes)
    colors = "#996888"
    if node_categories:
        categories = _get_node_info(graph_nodes, node_categories)
        colors = [category_colors[c] for c in categories]
    G.add_nodes_from(graph_node_names)
    graphs_per_edge_class = {k:G.copy() for k in edge_classes}
    
    nodes_from, nodes_to, edge_indices = np.argwhere(graph_edges > EDGE_THRESH)

    for n_from, n_to, edg_idx in zip(list(nodes_from), list(nodes_to), list(edge_indices)):
        G.add_edge(graph_node_names[n_from], graph_node_names[n_to])
        graphs_per_edge_class[edge_classes[edg_idx]].add_edge(graph_node_names[n_from], graph_node_names[n_to], weight=float(graph_edges[n_from][n_to][edg_idx]))
    
    return _draw_graph(G, graphs_per_edge_class, edge_classes, axs=axs, pos=pos, colors=colors)


def visualize_datapoint(model, dataloader, node_classes, edge_classes, node_categories = []):
    data_list = list(dataloader)
    inp = input('Do you want to visualize any output? (y/n)')

    while(inp == 'y' and len(data_list)>0):
        data = data_list.pop()
        edges = data['edges']
        nodes = data['nodes']
        context = data['context']
        y = data['y']

        x_hat = model.forward(edges, nodes, context)
        if SINGLE_PLOT:
            fig, single_axs = plt.subplots(1,3)
            axs = np.tile(single_axs.reshape(3,1),(1,len(edge_classes)))
        else:
            fig, axs = plt.subplots(3,len(edge_classes))
            axs = axs.reshape(3,len(edge_classes))
        positions = _visualize_graph(nodes.squeeze(0), edges.squeeze(0), node_classes, edge_classes, axs = axs[0,:], node_categories=node_categories)
        axs[0,0].set_ylabel('Input ')
        _visualize_graph(nodes.squeeze(0), x_hat.squeeze(0), node_classes, edge_classes, axs = axs[1,:], pos=positions, node_categories=node_categories)
        axs[1,0].set_ylabel('Predicted')
        _visualize_graph(nodes.squeeze(0), y.squeeze(0), node_classes, edge_classes, axs = axs[2,:], pos=positions, node_categories=node_categories)
        axs[2,0].set_ylabel('Actual')
        loss = float(model.losses(x_hat, y, nodes).mean())
        fig.suptitle('Context : '+str(list(context))+'; Loss : '+str(loss))
        
        plt.show()
        # plt.savefig('temp.jpg')
        inp = input('Do you want to visualize another output? (y/n)')



def visualize_routine(routine, dt=10, sparsify=True, remove_close = True):
    graphs = routine['graphs']
    times = routine['times']
    edge_classes = ["INSIDE", "ON", "CLOSE"]
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
        edge_labels = {}
        for edge in graph['edges']:
            if remove_close and edge['relation_type'] == "CLOSE":
                continue
            n1, n2 = edge['from_id'], edge['to_id']
            G.add_edge(n1, n2, weight=1)
            edge_labels[(n1,n2)] = edge['relation_type']
        
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
        nx.draw_networkx_edge_labels(G, pos=pos, ax=axs[i], edge_labels=edge_labels)
        th = time_external(t*dt)
        axs[i].set_title(str(int(th[2]))+':'+str(int(th[3])))
    th = [int(t) for t in th]
    fig.suptitle("Week "+str(th[0])+" - "+days[th[1]])
    plt.show()
    
