import networkx as nx
from networkx.classes import graph
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

EDGE_THRESH = 0.1
SEED = 456
SINGLE_PLOT = False

def _get_node_list(graph_nodes, node_classes):
    node_names = []
    for node in list(graph_nodes):
        node_names.append(node_classes[node.argmax()])
    return node_names

def _visualize_graph(graph_nodes, graph_edges, node_classes, edge_classes, axs=None, pos=None):
    G = nx.DiGraph()
    graph_node_names = _get_node_list(graph_nodes, node_classes)
    G.add_nodes_from(graph_node_names)
    edge_class_graphs = {k:G.copy() for k in edge_classes}
    
    nodes_from, nodes_to, edge_indices = np.argwhere(graph_edges > EDGE_THRESH)

    for n_from, n_to, edg_idx in zip(list(nodes_from), list(nodes_to), list(edge_indices)):
        G.add_edge(graph_node_names[n_from], graph_node_names[n_to])
        edge_class_graphs[edge_classes[edg_idx]].add_edge(graph_node_names[n_from], graph_node_names[n_to], weight=float(graph_edges[n_from][n_to][edg_idx]))
    
    if axs is None:
        _, axs = plt.subplots(1,len(edge_classes))
    
    if pos is None:
        pos = nx.spring_layout(G)
    for ax,edge_cl in zip(axs, edge_classes):
        if edge_cl == "CLOSE" and SINGLE_PLOT:
            continue
        weights = [edge_class_graphs[edge_cl][u][v]['weight'] for u,v in edge_class_graphs[edge_cl].edges()]
        nx.draw_networkx(G, pos=pos, ax=ax, edge_cmap=plt.cm.Blues, edge_vmin = 0 ,edge_vmax = 1 , edgelist=edge_class_graphs[edge_cl].edges(), edge_color=weights, node_size=1000, node_color='#C4EBC8')
        nx.draw_networkx_edge_labels(G, pos=pos, ax=ax, edge_labels={k:edge_cl for k in edge_class_graphs[edge_cl].edges()})
        ax.set_title(edge_cl)
    return pos

def visualize_datapoint(model, dataloader, node_classes, edge_classes):
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
        positions = _visualize_graph(nodes.squeeze(0), edges.squeeze(0), node_classes, edge_classes, axs = axs[0,:])
        axs[0,0].set_ylabel('Input ')
        _visualize_graph(nodes.squeeze(0), x_hat.squeeze(0), node_classes, edge_classes, axs = axs[1,:], pos=positions)
        axs[1,0].set_ylabel('Predicted')
        _visualize_graph(nodes.squeeze(0), y.squeeze(0), node_classes, edge_classes, axs = axs[2,:], pos=positions)
        axs[2,0].set_ylabel('Actual')
        fig.suptitle(str(context))
        
        plt.show()
        # plt.savefig('temp.jpg')
        inp = input('Do you want to visualize another output? (y/n)')
