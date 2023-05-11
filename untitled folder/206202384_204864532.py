import numpy as np
import networkx as nx
import random
import pandas as pd

df_instaglam_1 = pd.read_csv("instaglam_1.csv")
df_instaglam0 = pd.read_csv("instaglam0.csv")
df_spotifly = pd.read_csv("spotifly.csv")

G_1 = nx.from_pandas_edgelist(df_instaglam_1, "userID", "friendID")

counter = 0
counter_iteration = 0

def prob_for_edge(G, node1, node2):
    node1_num_neighboors = G.degree[node1]
    node2_num_neighboors = G.degree[node2]
    common_neighboors = len(list(nx.common_neighbors(G, node1, node2)))
    if common_neighboors <= 1:
        return 0
    p = common_neighboors / (node1_num_neighboors + node2_num_neighboors)
    return p


def updated_graph(G):
    global counter_iteration
    random_number = random.random()
    graph_nodes_list = list(G.nodes)
    graph_nodes_np_list = np.array(graph_nodes_list)
    unique_graph_nodes_np_list = np.unique(graph_nodes_np_list)
    n = len(unique_graph_nodes_np_list)
    print(counter_iteration)
    for i in range(n):
        for j in range(i + 1, n):
            if G.has_edge(unique_graph_nodes_np_list[i], unique_graph_nodes_np_list[j]):
                continue
            p = prob_for_edge(G, unique_graph_nodes_np_list[i], unique_graph_nodes_np_list[j])
            if random_number <= p:
               G.add_edge(unique_graph_nodes_np_list[i], unique_graph_nodes_np_list[j])



    for node in G.nodes:
        if len(G.adj[node]) >= 20:
            lst = np.random.choice(unique_graph_nodes_np_list, 20)
        elif len(G.adj[node]) >= 10:
            lst = np.random.choice(unique_graph_nodes_np_list, 10)
        elif len(G.adj[node]) >= 5:
            lst = np.random.choice(unique_graph_nodes_np_list, 5)
        else:
             lst = []
        if lst != [] :
            for neigh in lst:
                G.add_edge(node, neigh)



def h_update(G, artist):
    for node in G.nodes:
        plays_as_np_array = ((df_spotifly[(df_spotifly.userID == node) &
                                          (df_spotifly[' artistID'] == artist)]['#plays'])).to_numpy()
        if plays_as_np_array.size == 0:
            G.nodes[node]['h'] = 0
        else:
            G.nodes[node]['h'] = plays_as_np_array[0]


def parameters_update(G):
    for node in G.nodes:
        if G.nodes[node]['bought'] == 1:
            continue
        G.nodes[node]['Nt'] = len(G[node])
        for neighboor in G.adj[node]:
            if G.nodes[neighboor]['bought'] == 1:
                G.nodes[node]['Bt'] += 1
        if G.nodes[node]['h'] == 0:
            G.nodes[node]['probability'] = G.nodes[node]['Bt'] / G.nodes[node]['Nt']

        else:
            G.nodes[node]['probability'] = (G.nodes[node]['Bt'] * G.nodes[node]['h']) \
                                           / (1000 * G.nodes[node]['Nt'])


def buying_probability(G):
    global counter
    random_number = random.random()
    for node in G.nodes:
        if G.nodes[node]["bought"] == 1:
            continue
        x = G.nodes[node]['probability']
        if random_number <= x:
            G.nodes[node]['bought'] = 1
            counter += 1


def find_potential_influencers(G0):
    closeness = nx.closeness_centrality(G0, u=None, distance=None, wf_improved=True)
    max_closeness = dict(sorted(closeness.items(), key=lambda item: item[1], reverse=True))

    data_items1 = max_closeness.items()
    data_list1 = list(data_items1)
    max_closeness_df = pd.DataFrame(data_list1)
    max_closeness_df.rename(columns={0: 'UserID', 1: 'Closeness'}, inplace=True)
    max_closeness_df = max_closeness_df.drop(labels=range(20, 1892), axis=0)

    betweenness = nx.betweenness_centrality(G0, k=None, normalized=True, weight=None, endpoints=False, seed=None)
    max_betweenness = dict(sorted(betweenness.items(), key=lambda item: item[1], reverse=True))

    data_items2 = max_betweenness.items()
    data_list2 = list(data_items2)
    max_betweenness_df = pd.DataFrame(data_list2)
    max_betweenness_df.rename(columns={0: 'UserID', 1: 'Betweenness'}, inplace=True)
    max_betweenness_df = max_betweenness_df.drop(labels=range(20, 1892), axis=0)

    harmonic = nx.harmonic_centrality(G0, nbunch=None, distance=None, sources=None)
    max_harmonic = dict(sorted(harmonic.items(), key=lambda item: item[1], reverse=True))

    data_items3 = max_harmonic.items()
    data_list3 = list(data_items3)
    max_harmonic_df = pd.DataFrame(data_list3)
    max_harmonic_df.rename(columns={0: 'UserID', 1: 'Harmonic'}, inplace=True)
    max_harmonic_df = max_harmonic_df.drop(labels=range(20, 1892), axis=0)

    dict1 = {}
    for i in max_closeness_df['UserID']:
        dict1[i] = 1

    for i in max_betweenness_df['UserID']:
        if i not in dict1:
            dict1[i] = 1
        else:
            dict1[i] += 1

    for i in max_harmonic_df['UserID']:
        if i not in dict1:
            dict1[i] = 1
        else:
            dict1[i] += 1

    dict1 = dict(sorted(dict1.items(), key=lambda item: item[1], reverse=True))
    list_potential_influencers = list(dict1.keys())
    list_potential_influencers = list_potential_influencers[0:10]  # all the nodes that appeared in the top 30
                                                                   # of closeness, betweenness and harmonic measure
    return list_potential_influencers


def hill_climb(artist, all_options_influencers):
    global counter
    n = len(all_options_influencers)
    influencers = []
    k = 0
    while k != 5:
        dict1 = {}
        for i in range(n):
            if k == 0:
                G = reset_G(all_options_influencers[i], k)
                h_update(G, artist)
                dict1[all_options_influencers[i]] = simulation(G)
                counter = 0
                print(dict1)
            else:
                temp = influencers + [all_options_influencers[i]]
                G = reset_G(temp, k)
                h_update(G, artist)
                dict1[tuple(temp)] = simulation(G)
                print(dict1)
        max_node = max(dict1, key=dict1.get)
        if k == 0:
            influencers.append(max_node)
            all_options_influencers.remove(max_node)
        else:
            influencers.append(max_node[-1])
            all_options_influencers.remove(max_node[-1])
        n = len(all_options_influencers)
        k += 1
    return influencers


def reset_G(influencers, k):
    G0 = nx.from_pandas_edgelist(df_instaglam0, "userID", "friendID")
    nx.set_node_attributes(G0, 0, name="bought")
    nx.set_node_attributes(G0, 0, name="probability")
    nx.set_node_attributes(G0, 0, name="Nt")
    nx.set_node_attributes(G0, 0, name="Bt")
    nx.set_node_attributes(G0, 0, name="h")

    if k == 0:
        G0.nodes[influencers]['bought'] = 1

    else:
        for i in range(len(influencers)):
            G0.nodes[influencers[i]]['bought'] = 1
    return G0


def simulation(G):
    global counter
    for i in range(1, 7):
        updated_graph(G)
        parameters_update(G)
        buying_probability(G)
    return counter


def main():
    global counter
    artists = [989, 16326, 144882, 194647]
    G_start = nx.from_pandas_edgelist(df_instaglam0, "userID", "friendID")

    influencers_of_each_artists = []
    for i in range(4):
        print(f"loop for the {i+1} singer")
        all_options_influencers = find_potential_influencers(G_start)
        influencers_of_each_artists.append(hill_climb(artists[i], all_options_influencers))
        print(influencers_of_each_artists[i])
    print(influencers_of_each_artists)



if __name__ == '__main__':
    main()