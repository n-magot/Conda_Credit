import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

N_samples = 1000
nVars = 5
maxParents = 3
n_binary = 1

def Simulate_Mixed_Obs_data(nVars, maxParents, N_samples, n_binary):

    """function Simulate_Mixed_Obs_data(nVars, maxParents, N_samples, n_binary)
            Simulates data based on a dag. The dataset contains:
            1. One binary outcome
            2. One binary treatment assigned
            3. All the other variables will be binary or continuous

    Author: nandia.lelova@hotmail.com
    Inputs:
    1. nVars = number of variables
    2. maxParents = int: maximum number of parents per variable
    3. N_samples = number of simulated samples
    4. n_binary = number of binary variables except from the treatment and the outcome

    Outputs
    1. dag = adjacency matrix
    2. Dataframe created from the dag"""

    def randomdag(nVars, maxParents):
        '''
            Generate a random DAG over nVars variables where each variable has at most maxParents.
            The number of parents for each node is drawn uniformly from 1 to maxParents.
            Inputs:
                nVars = number of variables
                maxParents = int: maximum number of parents per variable
            Returns:
                dag = adjacency matrix
        '''

        dag = np.zeros((nVars, nVars))
        ordering = np.random.permutation(nVars)

        for iVar in range(1, nVars + 1):
            curVar = ordering[nVars - iVar]
            # maxParents +1 because with np.random.choice we have: e.g. np.choice(5) sample from 0,1,2,3,4
            nParents = min(np.random.choice(maxParents + 1), nVars - iVar)

            if nParents == 0:
                continue

            parents = np.random.choice(nVars - iVar, nParents, replace=False)
            dag[ordering[parents], curVar] = 1

        return dag


    def show_graph_with_labels(adjacency_matrix):
        """Show the Graph by giving the adjacency matrix"""
        rows, cols = np.where(adjacency_matrix == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.DiGraph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=500, with_labels=True)
        plt.show()


    def get_topo_order_nx(dag):
        '''
            Find the topological order of a DAG with networkx
            Args:
                dag:numpy array
                    adjency matrix
            Returns:
                topo_order: list
        '''

        G = nx.from_numpy_array(dag, create_using=nx.MultiDiGraph())
        topo_order = list(nx.topological_sort(G))

        return topo_order

    #Create a random DAG
    dag = randomdag(nVars, maxParents)

    #Get topological order of a dag
    topo = get_topo_order_nx(dag)

    """We want to create simulates for a binary outcome and a binary treatment. The outcome must be the last node not to
     have any children and at least one parent of an outcome must be binary.
     Adjacency's matrix information:
     1. Node without children: on his row has only 0
     2. At least one parent: on his column at least one 1
     """
    def nodes_without_children(dag):
        """ Input: adjacency matrix
            Output: Nodes without children"""

        zero_rows = np.all(dag == 0, axis=1)
        return np.where(zero_rows)[0]

    def nodes_with_at_least_one_parent(dag):
        """ Input: adjacency matrix
                Output: Nodes with at least one parent"""

        at_least_one_1 = np.any(dag == 1, axis=0)
        return np.where(at_least_one_1)[0]

    def find_Outcome(dag):
        """Find the node that doesn't have children and also has at least one parent.
        If there are more than one, randomly select one"""

        no_children = nodes_without_children(dag)
        one_parent = nodes_with_at_least_one_parent(dag)
        intersection = [element for element in no_children if element in one_parent]

        return np.random.choice(intersection)

    outcome = find_Outcome(dag)

    # Uncomment it for seeing the DAG's representation
    # show_graph_with_labels(dag)

    # Extract the parents of the outcome - Possible treatments
    treatment = np.random.choice(np.where(dag[:, outcome] == 1)[0])

    """Create an array that has value 1 in treatment and outcome indexes and either 0 for continuous variables or
     1 for binary variables based on how the variable "n_binary" selected. 0 and 1s assigned randomly
    """

    # Create a NumPy array of zeros
    typesVars = np.zeros(nVars, dtype=int)

    # Specify the indexes where you want to assign the value 1, the treatment and the outcome
    specific_indexes = [treatment, outcome]

    # Assign the value 1 to the specified indexes
    typesVars[specific_indexes] = 1

    # Assign the value 1 to n_binary additional indexes
    # Randomly select one additional index
    remaining_indexes = np.random.choice(np.setdiff1d(np.arange(nVars), specific_indexes), n_binary, replace=False)
    typesVars[remaining_indexes] = 1

    data = np.zeros((N_samples, nVars))
    for node in topo:
        #find if node has parents
        node_has_no_parents = not np.any(dag[:, node] == 1)
        if node_has_no_parents == True:
            #ean einai binary
            if typesVars[node] == 1:
                pr = np.random.uniform(0.1, 0.9, 1)
                data[:, node] = np.random.binomial(1, pr, size=N_samples)
            else:
                data[:, node] = np.random.normal(size=N_samples)
        else:
            #For every node in topological sort keep its parents and make coefs for its one
            parents_node = np.where(dag[:, node] == 1)[0]
            coefs = np.random.uniform(1.5, 3, size=len(parents_node))

            if typesVars[node] == 1:

                logit = np.sum(coefs * data[:, parents_node], axis=-1)
                pr = 1 / (1 + np.exp(-logit))
                data[:, node] = np.random.binomial(1, pr, size=N_samples)
            else:
                data[:, node] = np.sum(coefs * data[:, parents_node], axis=-1) + np.random.normal(scale=2, size=N_samples)

    """Create a dataframe with:
    1. The outcome in the first column
    2. The treatment in the second column"""
    df = pd.DataFrame(data)
    df = df.rename(columns={outcome: 'Outcome', treatment: 'Treatment'})

    df.insert(0, 'Outcome', df.pop('Outcome'))
    df.insert(1, 'Treatment', df.pop('Treatment'))

    return dag, df

dag, df = Simulate_Mixed_Obs_data(5,3,10, 1)


