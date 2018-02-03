
# coding: utf-8

# # Κώδικας για την εργασία κοινωνικών δικτύων - Κολιοπούλου, Κουτσιούμπη, Πανίδης

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


import networkx as nx
stackoverflow_data = pd.read_csv('../input/sx-stackoverflow.txt', sep=" ", header=None,
                                 names=["source_id", "target_id", "timestamp"])
# Take a sample of the data
stackoverflow_data.sort_values("timestamp")
end = int(len(stackoverflow_data)/3000)
data = stackoverflow_data[:end]

# User input for number of time periods and calculated range of each period
num = input("Enter number of time periods:")
n = int(num)
periodLength = int(len(data)/n)

# Creating the source parameter for nx.from_pandas_dataframe
g = {}
start = 0
for x in range(1, n+1):
    stop = periodLength * x
    g["s{0}".format(x)] = data[start:stop]
    start = stop

# initializing a dictionary for graphs
graphs = {}

# creating graphs based on user input
for x in range(1, n+1):
#     from_pandas_dataframe returns a graph from Pandas DataFrame.
    g["G{0}".format(x)] = nx.from_pandas_dataframe(g["s{0}".format(x)], 'source_id', 'target_id', 'timestamp', create_using=nx.DiGraph())
    


# In[2]:



import matplotlib.pyplot as plt
# create a graph using the columns source_id, target_id and timestamp from the file
for x in range(1, n+1):
    print("Graph G{0}".format(x))
    nx.draw(g["G{0}".format(x)])
    plt.show()


# In[3]:


# min timestamp

min = stackoverflow_data["timestamp"].min()
pd.to_datetime(min, unit='s')


# In[4]:


# max timestamp
max = stackoverflow_data["timestamp"].max()
pd.to_datetime(max, unit='s')


# In[5]:


# erotima 4
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib
# we initialize the dictionaries for each centrality measures
katz_centrality_dict = {}
degree_centrality_dict = {}
in_degree_centrality_dict = {}
out_degree_centrality_dict = {}
eigenvector_centrality_dict= {}
betweenness_centrality_dict = {}
closeness_centrality_dict = {}
dataframedict = {}
plt.rcParams.update({'font.size': 22})
for x in range(1, n+1):
#     katz centrality calculation
    katz_centrality_dict["G{0}".format(x)] = nx.katz_centrality_numpy(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(katz_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30, label='Katz Centrality')
#     plt.title('Katz Centrality Distribution for G{0}'.format(x))

#     degree centrality calculation
    degree_centrality_dict["G{0}".format(x)] = nx.degree_centrality(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(degree_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30,label='Degree Centrality')
#     plt.title('Degree Centrality Distribution for G{0}'.format(x))
    
#     in-degree centrality calculation
    in_degree_centrality_dict["G{0}".format(x)] = nx.out_degree_centrality(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(in_degree_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30, label='In-Degree Centrality')
#     plt.title('In-Degree Centrality Distribution for G{0}'.format(x))

#     out centrality calculation
    out_degree_centrality_dict["G{0}".format(x)] = nx.out_degree_centrality(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(out_degree_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30, label='Out-Degree Centrality')
    plt.title('Out-degree Centrality Distribution for G{0}'.format(x))
    
#     eigenvector centrality calculation
    eigenvector_centrality_dict["G{0}".format(x)] = nx.eigenvector_centrality_numpy(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(eigenvector_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30, label='Eigenvector Centrality')
#     plt.title('Eigenvector Centrality Distribution for G{0}'.format(x))

#     betweenness centrality calculation
    betweenness_centrality_dict["G{0}".format(x)] = nx.betweenness_centrality(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(betweenness_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30, label='Betweenness Centrality')
#     plt.title('Betweenness Centrality Distribution for G{0}'.format(x))
    
#     closeness centrality calculation
    closeness_centrality_dict["G{0}".format(x)] = nx.closeness_centrality(g["G{0}".format(x)])
    dataframedict["G{0}".format(x)] = pd.DataFrame(list(closeness_centrality_dict["G{0}".format(x)].items()), columns=['Node', 'degree_centrality'])
    plt.hist(dataframedict["G{0}".format(x)]['degree_centrality'], normed=True, bins=30, label='Closeness Centrality')
#     plt.title('Betweenness Centrality Distribution for G{0}'.format(x))    
    
    plt.title('Centrality Measures for G{0}'.format(x))
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(20.5, 12.5)
    leg = plt.legend()
#     plt.xticks(np.arange(0.0, 0.3, 0.02))
    plt.show()

    


# In[6]:


# ERWTIMA 5
intersection = {}
edges = {}
for x in range(1, n):
#     we find the nodes for each succesive subgraphs
 list1= g["G{0}".format(x)].nodes()
 
 list2= g["G{0}".format(x+1)].nodes()
# and the intersection (common nodes)
 intersection["V{0}".format(x)] = list(set(list1).intersection(list2))

 
 edges["E{0}V{1}".format(x,x)] = []
 edges["E{0}V{1}".format(x+1,x)] = []
# for each succesive graph we find the common nodes and therefore the common edges
 for i in intersection["V{0}".format(x)]:
     for a,v in g["G{0}".format(x)].edges(i):
         edges["E{0}V{1}".format(x,x)].append((a,v))

 for i in intersection["V{0}".format(x)]:
     for a,v in g["G{0}".format(x+1)].edges(i):
         edges["E{0}V{1}".format(x+1,x)].append((a,v))


# In[7]:


# erwtima 6
shortest_paths= {}
g_test = {}
for x in range(1, n):

    g_test["E{0}V{1}".format(x,x)]=nx.Graph()
#     we fill an empty graph with nodes as the edges from the previus step, and the according nodes
    for a,b in edges["E{0}V{1}".format(x,x)]:
        g_test["E{0}V{1}".format(x,x)].add_nodes_from([a,b])
        g_test["E{0}V{1}".format(x,x)].add_edge(a,b)

    shortest_paths["V{0}".format(x)]=[]
    for c in intersection["V{0}".format(x)]:
        for d in intersection["V{0}".format(x)]:
            if(c in g_test["E{0}V{1}".format(x,x)].nodes() and d in g_test["E{0}V{1}".format(x,x)].nodes()):
                if(nx.has_path(g_test["E{0}V{1}".format(x,x)], source = c,target = d)):
                    shortest_paths["V{0}".format(x)].append([c,d,nx.shortest_path_length(g_test["E{0}V{1}".format(x,x)], source = c,target = d)])


    


# In[8]:


common_neighbors = {}
for x in range(1, n):
    common_neighbors["V{0}".format(x)]=[]
    for c in intersection["V{0}".format(x)]:
        for d in intersection["V{0}".format(x)]:
            if(c in g_test["E{0}V{1}".format(x,x)].nodes() and d in g_test["E{0}V{1}".format(x,x)].nodes()):
                common_neighbors["V{0}".format(x)].append([c,d,len(sorted(nx.common_neighbors(g_test["E{0}V{1}".format(x,x)], c, d)))])


# In[9]:


jaccard_coef = {}
for x in range(1, n):
    jaccard_coef["V{0}".format(x)]=[]
    for c in intersection["V{0}".format(x)]:
        for d in intersection["V{0}".format(x)]:
            if(c in g_test["E{0}V{1}".format(x,x)].nodes() and d in g_test["E{0}V{1}".format(x,x)].nodes()):
                temp = nx.jaccard_coefficient(g_test["E{0}V{1}".format(x,x)],[(c,d)])
                for u, v, p in temp:
                    jaccard_coef["V{0}".format(x)].append([u,v,p])


# In[10]:


from math import log
adamic_adar = {}
ok = False
for x in range(1, n):
    adamic_adar["V{0}".format(x)] = []
    for c in intersection["V{0}".format(x)]:
        for d in intersection["V{0}".format(x)]:
#             checking that c and d nodes from the intersection exist in the graph of edges
            if(c in g_test["E{0}V{1}".format(x,x)].nodes() and d in g_test["E{0}V{1}".format(x,x)].nodes()):
#                 checking that common neighbors of the two nodes are more than one so that we avoid the error of division by 0
                if(len(sorted(nx.common_neighbors(g_test["E{0}V{1}".format(x,x)], c, d))) > 1):
#      iteration of all the common neighbors of the two nodes so that we check if one of them has degree less than two, so that we avoid the division by 0 error (log(0) or log(1))
                    for w in sorted(nx.common_neighbors(g_test["E{0}V{1}".format(x,x)], c, d)):
                        if(g_test["E{0}V{1}".format(x,x)].degree(w) <= 1):
                            ok = False;
                            break;
                        else:
                            ok = True;
                    if ok:
#                         if no degrees are 0 or 1 then we run the algorithm
                        temp = nx.adamic_adar_index(g_test["E{0}V{1}".format(x,x)],[(c,d)])
                        for u, v, p in temp:
                            adamic_adar["V{0}".format(x)].append([u,v,p])


# In[11]:


preferential_attachment = {}
for x in range(1, n):
    preferential_attachment["V{0}".format(x)] = []
    for c in intersection["V{0}".format(x)]:
        for d in intersection["V{0}".format(x)]:
            if(c in g_test["E{0}V{1}".format(x,x)].nodes() and d in g_test["E{0}V{1}".format(x,x)].nodes()):
                temp = nx.preferential_attachment(g_test["E{0}V{1}".format(x,x)],[(c,d)])
                
                for u, v, p in temp:
                    preferential_attachment["V{0}".format(x)].append([u, v, p])


# In[12]:


# erotima 7
input2 = input("Enter percentage:")
input_2 = int(input2)

jaccard_success = {}
preferential_success = {}
adamic_success = {}
common_success = {}
shortest_success = {}

for x in range(1, n):
#     print(common_neighbors["V{0}".format(x)])
#     creating the dataframes from the dictionary of metrics
    df_jaccard = pd.DataFrame(jaccard_coef["V{0}".format(x)],columns=['source','target','metric'])
    df_preferential_attachment = pd.DataFrame(preferential_attachment["V{0}".format(x)],columns=['source','target','metric'])
    df_adamic_adar = pd.DataFrame(adamic_adar["V{0}".format(x)],columns=['source','target','metric'])
    df_common_neighbors = pd.DataFrame(common_neighbors["V{0}".format(x)],columns=['source','target','metric'])
    df_shortest_paths = pd.DataFrame(shortest_paths["V{0}".format(x)],columns=['source','target','metric'])
#     sorting the dataframes we created according to the metric value in descending form
    df_jaccard_sorted = df_jaccard.sort_values(by=['metric'],ascending=False)
    df_preferential_attachment_sorted = df_preferential_attachment.sort_values(by=['metric'],ascending=False)
    df_adamic_adar_sorted = df_adamic_adar.sort_values(by=['metric'],ascending=False)
    df_common_neighbors_sorted = df_common_neighbors.sort_values(by=['metric'],ascending=False)
    df_shortest_paths_sorted = df_shortest_paths.sort_values(by=['metric'],ascending=False)
#     extracting the size of the dataframe according to the percentage given by user
    input1 = int((input_2/100) * len(df_jaccard_sorted))
    df_jaccard_sorted_with_percentage = df_jaccard_sorted[:input1]
    input1 = int((input_2/100) * len(df_preferential_attachment_sorted))
    df_preferential_attachment_sorted_with_percentage = df_preferential_attachment_sorted[:input1]
    input1 = int((input_2/100) * len(df_adamic_adar_sorted))
    df_adamic_adar_sorted_with_percentage = df_adamic_adar_sorted[:input1]
    input1 = int((input_2/100) * len(df_common_neighbors_sorted))
    df_common_neighbors_sorted_with_percentage = df_common_neighbors_sorted[:input1]
    input1 = int((input_2/100) * len(df_shortest_paths_sorted))
    df_shortest_paths_sorted_with_percentage = df_shortest_paths_sorted[:input1]
#     converting the previous dataframe to list
    jaccard_list = df_jaccard_sorted_with_percentage.values.tolist()
    preferential_list = df_preferential_attachment_sorted_with_percentage.values.tolist()
    adamic_list = df_adamic_adar_sorted_with_percentage.values.tolist()
    common_list = df_common_neighbors_sorted_with_percentage.values.tolist()
    shortest_list = df_shortest_paths_sorted_with_percentage.values.tolist()
#     calculating the success for each measure
    counter = 0
    for a,b,c in jaccard_list:
        for u,v in edges["E{0}V{1}".format(x+1,x)]:
            if(int(a)==u and int(b)==v):
                counter +=1
    jaccard_success["E{0}V{1}".format(x+1,x)] = (counter/len(edges["E{0}V{1}".format(x+1,x)]))
    counter = 0
    for a,b,c in preferential_list:
        for u,v in edges["E{0}V{1}".format(x+1,x)]:
            if(int(a)==u and int(b)==v):
                counter +=1    
    preferential_success["E{0}V{1}".format(x+1,x)] = (counter/len(edges["E{0}V{1}".format(x+1,x)]))
    counter = 0
    for a,b,c in adamic_list:
        for u,v in edges["E{0}V{1}".format(x+1,x)]:
            if(int(a)==u and int(b)==v):
                counter +=1    
    adamic_success["E{0}V{1}".format(x+1,x)] = (counter/len(edges["E{0}V{1}".format(x+1,x)]))
    counter = 0
    for a,b,c in common_list:
        for u,v in edges["E{0}V{1}".format(x+1,x)]:
            if(int(a)==u and int(b)==v):
                counter +=1    
    common_success["E{0}V{1}".format(x+1,x)] = (counter/len(edges["E{0}V{1}".format(x+1,x)]))
    counter = 0
    for a,b,c in shortest_list:
        for u,v in edges["E{0}V{1}".format(x+1,x)]:
            if(int(a)==u and int(b)==v):
                counter +=1    
    shortest_success["E{0}V{1}".format(x+1,x)] = (counter/len(edges["E{0}V{1}".format(x+1,x)]))
    counter = 0
                                                      
        
print("jaccard_success",jaccard_success)
print("preferential_success",preferential_success)
print("adamic_success",adamic_success)
print("common_success",common_success)
print("shortest_success",shortest_success)
    

