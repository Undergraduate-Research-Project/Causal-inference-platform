import numpy as np
from itertools import product

from seaborn.external.docscrape import header
from tensorflow.python.ops.distributions.kullback_leibler import kl_divergence

from .utilities import *
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import wasserstein_distance_nd


class EffectiveInformation:

    def __init__(self, data_file_path, G: nx.DiGraph,
                 binning_num=5, grade_name="grade"):
        self._data = pd.read_csv(data_file_path)
        self._target_name = None
        self._treatment_name = None
        self._confounder_names = None
        self._f_do = None
        self._corr_mat = None
        self._mi_mat = None
        self.BINNING_NUM = binning_num
        # 这个变量相关的内容都是写死的，需要未来改善
        self.GRADE_NAME = grade_name
        self.GRADE_NUM = 6
        self._data_binning()
        self._causality_graph = G

        # 1. 提取出因果图中的所有节点名称
        graph_nodes = set(self._causality_graph.nodes())

        # 2. 找到 self._data 中与因果图节点对应的列
        data_columns = set(self._data.columns)
        matching_columns = graph_nodes.intersection(data_columns)

        # 3. 删除 self._data 中没有对应列的节点
        self._data = self._data[list(matching_columns)]

        # 4.计算需要使用的矩阵
        self._compute_matrices()

    # data discretization
    def _data_binning(self):
        for col in self._data:
            # 这句写死了，可能日后需要处理
            if col == self.GRADE_NAME:
                continue
            # labels=False 是为了把把桶哪一行用标签换掉，不然就会出一个范围
            self._data[col] = pd.cut(self._data[col], self.BINNING_NUM, labels=False)

    def get_all_vars(self):

        return [str(node) for node in self._causality_graph.nodes()]


    def set_var(self, treatment_name: str, target_name: str):
        # self.BINNING_NUM = binning_num
        if not nx.has_path(self._causality_graph, treatment_name, target_name):
            return False
        self._target_name = target_name
        self._treatment_name = treatment_name
        confounder_names = find_adjustment_vars(self._causality_graph, treatment_name, target_name)
        self._confounder_names = confounder_names
        self._generate_f_do([target_name] + confounder_names + [treatment_name],
                            self._confounder_names + [treatment_name])
        return True
    def _generate_f_do(self, all_var_names, condition_names):
        # 第一步，根据给定变量，统计产生三个联合概率
        # 第二步，产生计算结果
        # 第三步，根据给定变量求和
        # joint_prob = (self._data.groupby(all_var_names).size()
        #               .div(len(self._data)).reset_index(name='P(y,t,c)'))
        # margin_prob = (self._data.groupby(condition_names).size()
        #                .div(len(self._data)).reset_index(name='P(t,c)'))
        # # 判断 confounder_names 是否为空
        # if self._confounder_names:
        #     # 如果不为空，按 confounder_names 分组计算 confounder_prob
        #     confounder_prob = (self._data.groupby(self._confounder_names).size()
        #                        .div(len(self._data)).reset_index(name='P(c)'))
        # else:
        #     # 如果为空，创建一个全为1的列 'P(c)'，长度与 margin_prob 相同
        #     confounder_prob = pd.DataFrame({'P(c)': [1] * len(margin_prob)})
        #
        #
        # # 合并上述结果
        # merged_temp = pd.merge(joint_prob, margin_prob)
        # merged = pd.merge(merged_temp, confounder_prob, left_index=True, right_index=True)
        # merged['P(y|t,c)*P(c)'] = (merged['P(y,t,c)'] / (merged['P(t,c)'])) * merged['P(c)']
        #
        # # 此处，只考虑了单个target和单个treatment，有需要的话可能得改（全都要改）
        # all_possible_input = list(product(range(self.GRADE_NUM),range(self.BINNING_NUM)))
        # new_rows = []
        # new_indices = []
        # # 技巧：这里要从表中检索所有符合情况的求和，而不是穷举所有情况去表中查找，不然超级慢
        # for (target, treatment) in all_possible_input:
        #     filter = {
        #         self._treatment_name: treatment,
        #         self._target_name: target
        #     }
        #
        #     filtered_df = merged.copy()
        #     for column, value in filter.items():
        #         filtered_df = filtered_df[filtered_df[column] == value]
        #     sum_of_column = filtered_df['P(y|t,c)*P(c)'].sum()
        #
        #     new_rows.append({"f_do": sum_of_column})
        #     new_indices.append((target, treatment))


            # Step 1: 计算联合频数
        ytc = self._data[all_var_names].value_counts().rename('ytc')
        tc = self._data[condition_names].value_counts().rename('tc')

        # 处理 confounder_names 为空的情况
        if self._confounder_names:
            c = self._data[self._confounder_names].value_counts().rename('c')
        else:
            c = pd.Series(len(self._data), name='c')  # 作为标量处理

        # 合并频数表
        merged = ytc.reset_index().merge(tc.reset_index(), on=condition_names, how='left')

        if self._confounder_names:
            merged = merged.merge(c.reset_index(), on=self._confounder_names, how='left')
        else:
            merged['c'] = c.iloc[0]  # 标量值填充

        total = len(self._data)
        merged['prob'] = (merged['ytc'] / merged['tc']) * (merged['c'] / total)

        # 提取y和t变量
        y_var = all_var_names[0]
        t_vars = [var for var in condition_names if var not in self._confounder_names]

        # 按y和t分组求和
        f_do_sum = merged.groupby([y_var] + t_vars)['prob'].sum().reset_index()

        # 构建MultiIndex DataFrame
        new_indices = list(f_do_sum[[y_var] + t_vars].itertuples(index=False, name=None))
        new_rows = f_do_sum['prob'].values.tolist()

        # print(f"\nDEBUG: SUM OF {sum(new_rows)}\n")
        self._f_do = pd.DataFrame(new_rows)
        self._f_do.index = pd.MultiIndex.from_tuples(new_indices, names=["target", "treatment"])

    def get_f_do(self, target: int, treatment: int):
        # out = self._f_do.loc[[(target, treatment)]]
        value = self._f_do.loc[(target, treatment)].values[0] if (target,treatment) in self._f_do.index else 0
        return value

    def measure_causal_effect(self):
        p, q = self.calculate_distributions()
        # print(p)
        kl = self.kl_divergence(p,q)
        js = self.js_divergence(p,q)
        total_var = self.total_variation_distance(p, q)
        wasserstein = self.wasserstein_distance(p, q)
        helligence = self.hellinger_distance(p, q)

        # Normalize both KL and JS by the number of bins
        # kl_divergence /= self.BINNING_NUM
        # js_divergence /= self.BINNING_NUM

        return [kl, js, total_var, wasserstein, helligence]

    def calculate_distributions(self):
        """
        Calculate the probability distributions for the target and treatment.
        Returns two lists: p (target distribution) and q (treatment distribution).
        """
        p = []
        q = []

        # 平均情况下的分布，作为被比较的对象，EI计算中分母的部分
        lst = []
        for target in range(self.GRADE_NUM):
            # Calculate the average distribution (denominator)
            denominator = np.mean([self.get_f_do(target, treatment) for treatment in range(self.BINNING_NUM)])
            lst.append(denominator)
        p = [lst] * self.BINNING_NUM


        for treatment in range(self.BINNING_NUM):
            lst = []
            for target in range(self.GRADE_NUM):
                lst.append(self.get_f_do(target, treatment))
            q.append(lst)

        p = np.array(p)
        q = np.array(q)
        # print(f"\nDEBUG: SUM OF Y|U {sum(sum(p))}\n")
        # print(f"DEBUG: SUM OF Y|do {sum(sum(q))}\n")

        return p, q

    # KL Divergence
    def kl_divergence(self, p, q):
        epsilon = 1e-10
        p = np.clip(p, epsilon, 1)
        q = np.clip(q, epsilon, 1)

        # 计算KL散度
        kl = np.sum(p * np.log(p / q))
        return kl

    # JS Divergence
    def js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * self.kl_divergence(p, m) + 0.5 * self.kl_divergence(q, m)

    # Total Variation Distance
    def total_variation_distance(self, p, q):
        return 0.5 * np.sum(np.abs(p - q))

    # Wasserstein Distance
    def wasserstein_distance(self, p, q):
        return wasserstein_distance_nd(p, q)

    # Hellinger Distance
    def hellinger_distance(self, p, q):
        return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


    def get_confounder(self):
        return self._confounder_names

    def _compute_matrices(self):
        # 计算相关系数矩阵
        self._corr_mat = self._data.corr()

        # 计算互信息矩阵
        mi_matrix = np.zeros((self._data.shape[1], self._data.shape[1]))
        for i, col_i in enumerate(self._data.columns):
            for j, col_j in enumerate(self._data.columns):
                if i != j:
                    # 互信息为对称矩阵，因此只计算一次
                    mi_matrix[i, j] = mutual_info_regression(
                        self._data[[col_i]], self._data[col_j]
                    )[0]
                    mi_matrix[j, i] = mi_matrix[i, j]

        # 存储互信息矩阵到 DataFrame，以便通过变量名称直接索引
        self._mi_mat = pd.DataFrame(
            mi_matrix, index=self._data.columns, columns=self._data.columns
        )

    def get_corr_value(self, var1: str, var2: str):
        return self._corr_mat.loc[var1, var2]

    def get_mi_value(self, var1: str, var2: str):
        return self._mi_mat.loc[var1, var2]

    def add_edge_weights(self, metric: str = "corr") -> nx.DiGraph:
        """
        生成 self._causality_graph 的拷贝图，并为每条边添加权重。

        :param metric: 字符串，控制选择互信息 ('mi') 还是相关系数 ('corr')，默认为 'corr'
        :return: 拷贝后的图，并附加了权重
        """
        # 复制causality_graph
        weighted_graph = self._causality_graph.copy()
        # 遍历每一条边
        for u, v in weighted_graph.edges():
            # 根据 metric 参数来决定使用互信息还是相关系数
            if metric == "mi":
                weight = self.get_mi_value(u, v)
            else:
                weight = self.get_corr_value(u, v)
            # 这里用了的绝对值，希望没问题
            weighted_graph[u][v]['weight'] = abs(weight)


        return weighted_graph

