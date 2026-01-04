import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class distance_on_performance():

    evaluation_matrix = np.array([])    # Matrix
    weighted_normalized = np.array([])  # Weight matrix
    normalized_decision = np.array([])  # Normalisation matrix

    M = 0  # Number of rows/employee
    N = 0  # Number of columns/criteria

    '''
	Create an evaluation matrix consisting of m alternatives and n criteria,
	with the intersection of each alternative and criteria given as {\displaystyle x_{ij}}x_{ij},
	we therefore have a matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}}.
	'''

    def __init__(self, evaluation_matrix, weight_matrix, criteria):
        # M×N matrix
        self.evaluation_matrix = np.array(evaluation_matrix, dtype="float")

        # M alternatives (options)
        self.row_size = len(self.evaluation_matrix)

        # N attributes/criteria
        self.column_size = len(self.evaluation_matrix[0])

        # N size weight matrix
        self.weight_matrix = np.array(weight_matrix, dtype="float")
        self.weight_matrix = self.weight_matrix/sum(self.weight_matrix)
        self.criteria = np.array(criteria, dtype="float")

    '''
	# Step 2
	The matrix {\displaystyle (x_{ij})_{m\times n}}(x_{{ij}})_{{m\times n}} is then normalised to form the matrix
	'''

    def step_2(self):
        # normalized scores
        self.normalized_decision = np.copy(self.evaluation_matrix)
        sqrd_sum = np.zeros(self.column_size)
        for i in range(self.row_size):
            for j in range(self.column_size):
                sqrd_sum[j] += self.evaluation_matrix[i, j]**2
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.normalized_decision[i,j] = self.evaluation_matrix[i, j]/(sqrd_sum[j]**0.5)

    '''
	# Step 3
	Calculate the weighted normalised decision matrix
	'''

    def step_3(self):
        from pdb import set_trace
        self.weighted_normalized = np.copy(self.normalized_decision)
        for i in range(self.row_size):
            for j in range(self.column_size):
                self.weighted_normalized[i, j] *= self.weight_matrix[j]

    '''
	# Step 4
	Determine the worst alternative {\displaystyle (A_{w})}(A_{w}) and the best alternative {\displaystyle (A_{b})}(A_{b}):
	'''

    def step_4(self):
        self.worst_alternatives = np.zeros(self.column_size)
        self.best_alternatives = np.zeros(self.column_size)
        for i in range(self.column_size):
            if self.criteria[i]:
                self.worst_alternatives[i] = min(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = max(self.weighted_normalized[:, i])
            else:
                self.worst_alternatives[i] = max(
                    self.weighted_normalized[:, i])
                self.best_alternatives[i] = min(self.weighted_normalized[:, i])

    '''
	# Step 5
	Calculate the L2-distance between the target alternative {\displaystyle i}i and the worst condition {\displaystyle A_{w}}A_{w}
	{\displaystyle d_{iw}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{wj})^{2}}},\quad i=1,2,\ldots ,m,}
	and the distance between the alternative {\displaystyle i}i and the best condition {\displaystyle A_{b}}A_b
	{\displaystyle d_{ib}={\sqrt {\sum _{j=1}^{n}(t_{ij}-t_{bj})^{2}}},\quad i=1,2,\ldots ,m}
	where {\displaystyle d_{iw}}d_{{iw}} and {\displaystyle d_{ib}}d_{{ib}} are L2-norm distances
	from the target alternative {\displaystyle i}i to the worst and best conditions, respectively.
	'''

    def step_5(self):
        self.worst_distance = np.zeros(self.row_size)
        self.best_distance = np.zeros(self.row_size)

        self.worst_distance_mat = np.copy(self.weighted_normalized)
        self.best_distance_mat = np.copy(self.weighted_normalized)

        for i in range(self.row_size):
            for j in range(self.column_size):
                self.worst_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.worst_alternatives[j])**2
                self.best_distance_mat[i][j] = (self.weighted_normalized[i][j]-self.best_alternatives[j])**2

                self.worst_distance[i] += self.worst_distance_mat[i][j]
                self.best_distance[i] += self.best_distance_mat[i][j]

        for i in range(self.row_size):
            self.worst_distance[i] = self.worst_distance[i]**0.5
            self.best_distance[i] = self.best_distance[i]**0.5

    '''
	# Step 6 Calculate the similarity or closeness to ideal solution
	'''

    def step_6(self):
        np.seterr(all='ignore')
        self.worst_similarity = np.zeros(self.row_size)
        self.best_similarity = np.zeros(self.row_size)

        for i in range(self.row_size):
            # calculate the similarity to the worst condition
            self.worst_similarity[i] = self.worst_distance[i] / (self.worst_distance[i] + self.best_distance[i])

            # calculate the similarity to the best condition
            self.best_similarity[i] = self.best_distance[i] / (self.worst_distance[i] + self.best_distance[i])

    def ranking(self, data):
        return [i+1 for i in data.argsort()]

    def rank_to_worst_similarity(self):
        # return rankdata(self.worst_similarity, method="min").astype(int)
        return self.ranking(self.worst_similarity)

    def rank_to_best_similarity(self):
        # return rankdata(self.best_similarity, method='min').astype(int)
        return self.ranking(self.best_similarity)

    def calc(self):
        print("Step 1: Raw data\n", self.evaluation_matrix, end="\n\n")
        self.step_2()
        print("Step 2: Normalized Matrix\n", self.normalized_decision, end="\n\n")
        self.step_3()
        print("Step 3: Calculate the weighted normalised decision matrix\n", self.weighted_normalized, end="\n\n")
        self.step_4()
        print("Step 4: Determine the worst alternative and the best alternative\n", self.worst_alternatives, self.best_alternatives, end="\n\n")
        self.step_5()
        print("Step 5: Distance from Best to Worst\n", self.worst_distance, self.best_distance, end="\n\n")
        self.step_6()
        print("Step 6: \n", self.worst_similarity, self.best_similarity, end="\n\n")

        closed_coefficient = []  # closeness coefficient

        for i in range(self.row_size):
            closed_coefficient.append(round(self.worst_similarity[i] / (self.best_similarity[i] + self.worst_similarity[i]), 3))

        q = [i + 1 for i in range(self.row_size)]

        plt.figure(figsize=(12, 6))  # Adjust the width and height as needed

        plt.plot(q, self.best_similarity, 'p--', color='red', markeredgewidth=2, markersize=8, label='Distance from the best')
        plt.plot(q, self.worst_similarity, '*--', color='blue', markeredgewidth=2, markersize=8, label='Distance from the worst')
        #plt.plot(q, closed_coefficient, 'o--', color='green', markeredgewidth=2, markersize=8, label='Close Co-Efficient')

        plt.title('Results')
        plt.xticks(range(self.row_size + 2))
        plt.axis([0, self.row_size + 1, 0, 1.2])
        plt.xlabel('Quality Members')
        plt.legend()
        plt.grid(True)
        plt.show()




def tier_rank(tier_labal, quantile, target_columns):
    results, bin_edges = pd.qcut(target_columns,
                            q= quantile,
                            labels= tier_labal,
                            retbins=True)

    tier_table = pd.DataFrame(zip(bin_edges, tier_labal),
                                columns=['Threshold', 'Tier'])
    
    return tier_table

def adjust_score(_max, _min, in_array):
    '''
    adjust the score to match the output rules of the company

    Parameters
    ----------
    _max : TYPE
        DESCRIPTION.
    _min : TYPE
        DESCRIPTION.
    array : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    input = in_array.copy()
    result = []

    for i in input:
        peer = (i - min(input)) / max(input)
        z = peer * (_max - _min) + _min
        result.append(z)

    return result

###############
def normalization_pms_toptalent(df, score_columns, weight_matrix, criteria):
    ''' 
    -- weight_matrix : list of weights corresponding to score_columns
    -- criteria : list of boolean values, True for criteria where higher is better, False otherwise
    '''
    
    # Extract the relevant quality scores from the DataFrame based on the provided columns
    quality = df[score_columns].values

    # Ensure the weights are normalized
    weight_matrix = np.array(weight_matrix, dtype="float")
    weight_matrix = weight_matrix / sum(weight_matrix)

    # Initialize the distance_on_performance class with dynamic score columns, weights, and criteria
    top = distance_on_performance(quality, weight_matrix, criteria)
    top.calc()
    
    print("best_distance\t", top.best_distance)
    print("worst_distance\t", top.worst_distance)

    print("weighted_normalized", top.weighted_normalized)

    print("worst_similarity\t", top.worst_similarity)
    print("rank_to_worst_similarity\t", top.rank_to_worst_similarity())

    print("best_similarity\t", top.best_similarity)
    print("rank_to_best_similarity\t", top.rank_to_best_similarity())

    members = df[['PERSON_ID']].values.flatten()

    raw = top.best_similarity
    _result = 1 - raw

    _max = 1
    _min = 0.3

    result = adjust_score(_max, _min, _result)
    print(result)
    print("\n")

    arr = [*sorted(zip(members, result), key=lambda x: x[1], reverse=True)]

    reg_df = pd.DataFrame(arr, columns=['PERSON_ID', 'TALENT_SCORE'])

    return reg_df

def toptalent(df, score_columns, weight_matrix, criteria, percentiles_list = [0, .40, .50, .70, .85, 1]):

    from sklearn.preprocessing import MinMaxScaler
    
    q = percentiles_list
    scaler = MinMaxScaler(feature_range=(0, 100))
    scaled_data = scaler.fit_transform(df[score_columns])

    scaled_df = pd.DataFrame(scaled_data, columns=[col + '_SCALED' for col in score_columns])

    final_selection_df = pd.concat([df, scaled_df], axis=1)

    reg_df = normalization_pms_toptalent(final_selection_df, [col + '_SCALED' for col in score_columns], weight_matrix, criteria)

    bin_labels = ['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
    # q = [0, .2, .4, .6, .8, .1] # Adjust the percentile

    tier_table = tier_rank(bin_labels, q, reg_df['TALENT_SCORE'])
    tier_table['Percentile'] = q[:-1]

    bins = list(tier_table['Threshold']) + [float('inf')]
    labels = tier_table['Tier']

    reg_df['TIER'] = pd.cut(reg_df['TALENT_SCORE'], bins=bins, labels=labels, right=False)

    toptalent = reg_df.copy(deep=True)
    tier_order = ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze']
    toptalent['TIER'] = pd.Categorical(toptalent['TIER'], categories=tier_order, ordered=True)
    toptalent = toptalent.sort_values('TIER')

    return toptalent, tier_table