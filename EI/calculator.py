import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import wasserstein_distance_nd
from .utilities import find_adjustment_vars

class EffectiveInformation:
    """
    A class to compute causal effect measurements (KL, JS, Total Variation,
    Wasserstein, Hellinger) between a target and treatment variable in a given
    causal graph using observational data.
    """

    def __init__(self, data_file_path, G: nx.DiGraph, binning_num=5):
        """
        Initialize with the data (CSV file) and causal graph.
        Parameters:
            data_file_path (str): Path to the CSV data file.
            G (nx.DiGraph): Directed graph of causal relationships.
            binning_num (int): Number of bins to use for numeric discretization.
        """
        # Load data
        self._data = pd.read_csv(data_file_path)
        self._target_name = None
        self._treatment_name = None
        self._confounder_names = None
        self._f_do = None
        self._corr_mat = None
        self._mi_mat = None
        self.BINNING_NUM = binning_num
        self._causality_graph = G

        # Filter dataset to columns present in the causal graph
        graph_nodes = set(self._causality_graph.nodes())
        data_columns = set(self._data.columns)
        matching_columns = list(graph_nodes.intersection(data_columns))
        self._data = self._data[matching_columns].copy()

        # Discretize numeric columns and encode categorical columns
        self._data_binning()

        # Compute correlation and mutual information matrices
        self._compute_matrices()

    def _data_binning(self):
        """
        Discretize continuous (numeric) variables into bins and encode
        categorical variables as numeric codes.
        """
        for col in self._data.columns:
            if pd.api.types.is_numeric_dtype(self._data[col]):
                # Continuous variable: discretize into equal-width bins
                self._data[col] = pd.cut(self._data[col],
                                         bins=self.BINNING_NUM,
                                         labels=False)
            else:
                # Categorical variable: convert to category codes
                self._data[col] = self._data[col].astype('category').cat.codes

    def get_all_vars(self):
        """
        Return all variable names in the causal graph.
        """
        return [str(node) for node in self._causality_graph.nodes()]

    def set_var(self, treatment_name: str, target_name: str):
        """
        Set the treatment and target variable names and compute the
        interventional distribution P(target | do(treatment)).
        Returns True if successful (treatment affects target), False otherwise.
        """
        # Check if there is a causal path from treatment to target
        if not nx.has_path(self._causality_graph, treatment_name, target_name):
            return False

        self._target_name = target_name
        self._treatment_name = treatment_name

        # Find confounders given the causal graph
        confounder_names = find_adjustment_vars(self._causality_graph,
                                                 treatment_name,
                                                 target_name)
        self._confounder_names = confounder_names

        # Compute the interventional distribution
        all_var_names = [target_name] + confounder_names + [treatment_name]
        condition_names = confounder_names + [treatment_name]
        self._generate_f_do(all_var_names, condition_names)
        return True

    def _generate_f_do(self, all_var_names, condition_names):
        """
        Compute the interventional probabilities P(y|do(t)) for target y and treatment t,
        possibly conditioning on confounders. The result is stored in self._f_do
        as a MultiIndex DataFrame indexed by (target, treatment).
        """
        # Count joint occurrences of (target, confounders..., treatment) and (confounders..., treatment)
        ytc = self._data[all_var_names].value_counts().rename('ytc')
        tc = self._data[condition_names].value_counts().rename('tc')

        # Count confounder occurrences
        if self._confounder_names:
            c = self._data[self._confounder_names].value_counts().rename('c')
        else:
            c = pd.Series(len(self._data), name='c')  # scalar count when no confounders

        # Merge counts: join ytc and tc on condition names
        merged = ytc.reset_index().merge(tc.reset_index(),
                                        on=condition_names, how='left')

        # Merge confounder counts if they exist
        if self._confounder_names:
            merged = merged.merge(c.reset_index(),
                                  on=self._confounder_names, how='left')
        else:
            # Fill with total count for confounder (no confounder case)
            merged['c'] = c.iloc[0]

        # Total number of samples
        total = len(self._data)
        # Compute joint probability * effect: P(y|t,c) * P(c)
        merged['prob'] = (merged['ytc'] / merged['tc']) * (merged['c'] / total)

        # Get target and treatment column names
        y_var = all_var_names[0]
        t_vars = [var for var in condition_names
                  if var not in (self._confounder_names or [])]

        # Sum over confounders to get P(y|do(t))
        f_do_sum = merged.groupby([y_var] + t_vars)['prob'].sum().reset_index()

        # Convert to MultiIndex for efficient lookups: index is (target, treatment)
        new_indices = list(f_do_sum[[y_var] + t_vars]
                           .itertuples(index=False, name=None))
        new_rows = f_do_sum['prob'].values.tolist()

        self._f_do = pd.DataFrame(new_rows)
        self._f_do.index = pd.MultiIndex.from_tuples(new_indices,
                                                     names=["target", "treatment"])

    def get_f_do(self, target, treatment):
        """
        Retrieve the interventional probability P(target_value | do(treatment_value)).
        Returns 0 if the combination is not present.
        """
        if (target, treatment) in self._f_do.index:
            return self._f_do.loc[(target, treatment)].values[0]
        else:
            return 0

    def calculate_distributions(self):
        """
        Construct probability distributions p (baseline) and q (interventional).
        Returns:
            p (ndarray): Baseline distribution where each row is identical (averaged over treatments).
            q (ndarray): Distribution under intervention do(treatment) for each treatment.
        """
        # Unique target and treatment values from the computed f_do
        target_values = sorted({t for t, _ in self._f_do.index})
        treatment_values = sorted({t for _, t in self._f_do.index})

        # Compute baseline probabilities: average over all treatments for each target
        baselines = []
        for y in target_values:
            vals = [self.get_f_do(y, t) for t in treatment_values]
            baselines.append(np.mean(vals))
        # Baseline distribution p: each row (treatment) is the same
        p = np.array([baselines for _ in treatment_values], dtype=float)

        # Interventional distribution q: each row corresponds to a treatment
        q = []
        for t in treatment_values:
            row = [self.get_f_do(y, t) for y in target_values]
            q.append(row)
        q = np.array(q, dtype=float)

        return p, q

    def measure_causal_effect(self):
        """
        Compute causal effect measures (KL, JS, Total Variation, Wasserstein, Hellinger)
        between the baseline and interventional distributions.
        Returns a list [KL, JS, TV, Wasserstein, Hellinger].
        """
        p, q = self.calculate_distributions()
        kl = self.kl_divergence(p, q)
        js = self.js_divergence(p, q)
        total_var = self.total_variation_distance(p, q)
        wasserstein = self.wasserstein_distance(p, q)
        hellinger = self.hellinger_distance(p, q)
        return [kl, js, total_var, wasserstein, hellinger]

    def kl_divergence(self, p, q):
        """
        Calculate the Kullback-Leibler (KL) divergence between distributions p and q.
        """
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)
        return np.sum(p * np.log(p / q))

    def js_divergence(self, p, q):
        """
        Calculate the Jensen-Shannon (JS) divergence between distributions p and q.
        """
        m = 0.5 * (p + q)
        return 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)

    def total_variation_distance(self, p, q):
        """
        Calculate the Total Variation distance between distributions p and q.
        """
        return 0.5 * np.sum(np.abs(p - q))

    def wasserstein_distance(self, p, q):
        """
        Calculate the (multi-dimensional) Wasserstein (Earth Mover's) distance between p and q.
        """
        return wasserstein_distance_nd(p, q)

    def hellinger_distance(self, p, q):
        """
        Calculate the Hellinger distance between distributions p and q.
        """
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))

    def get_confounder(self):
        """
        Return the list of confounder variable names identified.
        """
        return self._confounder_names

    def _compute_matrices(self):
        """
        Compute correlation and mutual information matrices for the dataset.
        """
        # Correlation matrix (Pearson)
        self._corr_mat = self._data.corr()

        # Mutual information matrix
        n = self._data.shape[1]
        mi_matrix = np.zeros((n, n))
        cols = list(self._data.columns)
        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                if i != j:
                    # Mutual information is symmetric, compute once and mirror
                    mi_val = mutual_info_regression(self._data[[col_i]],
                                                    self._data[col_j])[0]
                    mi_matrix[i, j] = mi_val
                    mi_matrix[j, i] = mi_val

        self._mi_mat = pd.DataFrame(mi_matrix, index=cols, columns=cols)

    def get_corr_value(self, var1: str, var2: str):
        """
        Return Pearson correlation coefficient between var1 and var2.
        """
        return self._corr_mat.loc[var1, var2]

    def get_mi_value(self, var1: str, var2: str):
        """
        Return mutual information between var1 and var2.
        """
        return self._mi_mat.loc[var1, var2]

    def add_edge_weights(self, metric: str = "corr") -> nx.DiGraph:
        """
        Return a copy of the causal graph with edge weights added based on the
        selected metric ('corr' or 'mi').
        Weights are the absolute value of the correlation or mutual information.
        """
        weighted_graph = self._causality_graph.copy()
        for u, v in weighted_graph.edges():
            if metric == "mi":
                weight = self.get_mi_value(u, v)
            else:
                weight = self.get_corr_value(u, v)
            weighted_graph[u][v]['weight'] = abs(weight)
        return weighted_graph
    


# 使用样例
# import networkx as nx

# # 创建因果图
# G = nx.DiGraph()

# # 添加因果关系边
# G.add_edges_from([
#     ("Rings", "Length"),
#     ("Rings", "Diameter"),
#     ("Rings", "Height"),
#     ("Length", "Shell weight"),
#     ("Diameter", "Shell weight"),
#     ("Height", "Shell weight"),
#     ("Shell weight", "Whole weight"),
#     ("Whole weight", "Shucked weight"),
#     ("Whole weight", "Viscera weight")
# ])

# # 检查是否无环（重要！）
# if nx.is_directed_acyclic_graph(G):
#     print("图验证通过：是有向无环图（DAG）")
# else:
#     print("错误：图中存在环！")



# from EI.calculator import *


# t = EffectiveInformation("test.csv",G)
# # 设置处理变量和目标变量
# success = t.set_var(
#     treatment_name="Shell weight", 
#     target_name="Whole weight"
# )

# if success:
#     # 计算因果效应指标
#     effects = t.measure_causal_effect()
#     print("KL散度:", effects[0])
#     print("JS散度:", effects[1])
# else:
#     print("因果路径不存在，请检查图结构！")

