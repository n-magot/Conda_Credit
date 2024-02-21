import numpy as np
import random
import networkx as nx

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
      """Show the Graph by giving the adjacency matrix as np.array"""
      rows, cols = np.where(adjacency_matrix == 1)
      edges = zip(rows.tolist(), cols.tolist())
      gr = nx.DiGraph()
      gr.add_edges_from(edges)
      nx.draw(gr, node_size=500, with_labels=True)
      plt.show()
