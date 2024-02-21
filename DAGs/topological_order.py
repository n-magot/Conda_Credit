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
