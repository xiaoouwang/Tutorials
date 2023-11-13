from scipy.stats import shapiro, ttest_rel, wilcoxon


class Compare:
    def __init__(self, scores_model_1, scores_model_2):
        self.scores_model_1 = scores_model_1
        self.scores_model_2 = scores_model_2

    def shapiro_test(self):
        shapiro_p_model_1 = shapiro(self.scores_model_1).pvalue
        shapiro_p_model_2 = shapiro(self.scores_model_2).pvalue
        return shapiro_p_model_1, shapiro_p_model_2

    def paired_t_test(self):
        t_stat, t_pvalue = ttest_rel(self.scores_model_1, self.scores_model_2)
        return t_stat, t_pvalue

    def wilcoxon_test(self):
        wilcoxon_stat, wilcoxon_pvalue = wilcoxon(self.scores_model_1, self.scores_model_2)
        return wilcoxon_stat, wilcoxon_pvalue

    def recommend_test(self):
        shapiro_p_model_1, shapiro_p_model_2 = self.shapiro_test()
        recommended_test = None
        if shapiro_p_model_1 > 0.05 and shapiro_p_model_2 > 0.05:
            recommended_test = 'Paired t-Test'
        else:
            recommended_test = 'Wilcoxon Signed-Rank Test'
        return recommended_test

# Example usage:
# Assuming you have two arrays of model F1 scores: f1_scores_model_1 and f1_scores_model_2
#

f1_scores_model_1 = [0.8, 0.82, 0.83, 0.81, 0.84]
f1_scores_model_2 = [0.79, 0.78, 0.80, 0.81, 0.79]

comparator = Compare(f1_scores_model_1, f1_scores_model_2)
shapiro_p1, shapiro_p2 = comparator.shapiro_test()
print("Shapiro-Wilk Test p-values:", shapiro_p1, shapiro_p2)

recommended_test = comparator.recommend_test()
print("Recommended Test:", recommended_test)

if recommended_test == 'Paired t-Test':
    t_stat, t_pvalue = comparator.paired_t_test()
    print("Paired t-Test p-value:", t_pvalue)
else:
    wilcoxon_stat, wilcoxon_pvalue = comparator.wilcoxon_test()
    print("Wilcoxon Signed-Rank Test p-value:", wilcoxon_pvalue)

