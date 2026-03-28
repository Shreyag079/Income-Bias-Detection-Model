import matplotlib.pyplot as plt

models = ["Baseline", "Preprocessing", "Inprocessing", "Postprocessing", "Counterfactual"]

accuracy = [0.845, 0.828, 0.805, 0.839, 0.845]
'''
plt.figure()
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("graphs/accuracy.png")   
plt.close()

dp = [0.170, 0.061, 0.313, 0.215, 0.166]

plt.figure()
plt.bar(models, dp)
plt.title("Demographic Parity Difference")
plt.xlabel("Models")
plt.ylabel("DP Difference (Lower is Better)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("graphs/demographic.png")   
plt.close()

eo = [0.068, 0.102, 0.181, 0.100, 0.061]

plt.figure()
plt.bar(models, eo)
plt.title("Equalized Odds Difference")
plt.xlabel("Models")
plt.ylabel("EO Difference (Lower is Better)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("graphs/eo.png")   
plt.close()

di = [0.347, 0.764, 0.314, 0.334, 0.357]

plt.figure()
plt.bar(models, di)
plt.title("Disparate Impact Comparison")
plt.xlabel("Models")
plt.ylabel("Disparate Impact (Closer to 1 is Better)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("graphs/di.png")   
plt.close()

cf = [0.098, 0.148, 0.128, 0.0]  
models_cf = ["Baseline", "Preprocessing", "Inprocessing", "Counterfactual"]

plt.figure()
plt.bar(models_cf, cf)
plt.title("Counterfactual Bias Comparison")
plt.xlabel("Models")
plt.ylabel("Bias Rate (Lower is Better)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("graphs/cf.png")   
plt.close()

groups = ["Male", "Female"]
ppr = [0.263, 0.081]

plt.figure()
plt.bar(groups, ppr)
plt.title("Positive Prediction Rate by Gender")
plt.xlabel("Group")
plt.ylabel("PPR")
plt.tight_layout()
plt.savefig("graphs/ppr.png")   
plt.close()

groups = ["Male", "Female"]
recall = [0.616, 0.522]

plt.figure()
plt.bar(groups, recall)
plt.title("Recall (TPR) by Gender")
plt.xlabel("Group")
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig("graphs/recall.png")   
plt.close()

groups = ["Male", "Female"]
fnr = [0.384, 0.478]

plt.figure()
plt.bar(groups, fnr)
plt.title("False Negative Rate (FNR) by Gender")
plt.xlabel("Gender")
plt.ylabel("FNR")
plt.tight_layout()

plt.savefig("graphs/fnr_gender.png")
plt.close()'''

dp = [0.170, 0.061, 0.313, 0.215, 0.166]

plt.figure()
plt.scatter(dp, accuracy)

for i, model in enumerate(models):
    plt.annotate(model, (dp[i], accuracy[i]))

plt.title("Accuracy vs Fairness Trade-off")
plt.xlabel("Demographic Parity Difference")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("graphs/tradeoff.png")
plt.close()