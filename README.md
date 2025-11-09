1 Abstract
Cardiovascular disease (CVD) is the leading cause of death worldwide, and early detection and management are crucial for individuals with CVD or at high risk. Therefore, researching the pathogenic factors of CVD is of great significance. To explore the pathogenic factors of cardiovascular disease, this code analyzed 1190 samples from four locations: Hungary, Switzerland, Cleveland, and Long Beach, Virginia. The analysis included various characteristics such as basic information, physiological indicators, symptoms, and test results. This code primarily uses a logit model for modeling, employing forward regression and lasso contraction estimation by comparing the AIC and BIC criteria to screen variables. Furthermore, adjusting thresholds improved the recall rate, contributing to the efficient screening of potential cardiovascular disease patients.

2 Missing Value Handling
Due to some screening subjects being unable to answer certain questions, the missing value rate in this dataset is relatively high, reaching 18.74%. Of the 918 samples, 172 samples were missing one variable (Cholesterol); one sample was missing two variables (Cholesterol and RestingBP); the remaining samples had complete data. To reduce model error, preserve as much information as possible, and simplify the missing value imputation process, this paper uses median imputation to fill in the 172 samples with a single missing value and removes the sample missing two variables.

3 Dependent Variable Handling
Due to the specific nature of disease diagnosis, we need to carefully consider each screening subject who may have cardiovascular disease. This sample contains 507 positive samples (with cardiovascular disease), accounting for 55.3%, and 410 negative samples (without cervical cancer), accounting for 44.7%, indicating a relatively balanced sample category ratio.

4 Model Selection

Due to strong autocorrelation among some variables and low correlation with cardiovascular disease, variable screening is necessary to remove variables with low correlation to the dependent variable and redundant variables. This aims to improve model accuracy and interpretability, and reduce runtime. The full model regression results are shown, with significant variables accounting for only 43.75%.

This analysis uses a logit model for regression modeling, employing forward regression as the variable selection method. The AIC and BIC criteria  are used, and the modeling results of the lasso contraction estimate under optimal parameters are compared. Simultaneously, the screening results of AIC and BIC, variable autocorrelation, and interpretability are considered.

To further compare the performance of the three models, this study also reports their AUC, number of variables, number and proportion of statistically significant variables. AUC is used to evaluate the model’s discriminative ability; the number of variables reflects model parsimony; and the number/proportion of significant variables helps assess potential multicollinearity or redundancy. Because the sample size is relatively small, 5-fold cross-validation was conducted, and the average residual sum of squares and average test error of each model are summarized below.

For clarity, the results corresponding to “Table 2: Model evaluation metrics” are described in text:

Full model:
AUC = 0.930
Average residual sum of squares = 17.0211
Average test error = 0.1086
Number of variables = 16
Number of significant variables = 7
Proportion of significant variables = 43.75%

AIC-selected model:
AUC = 0.930
Average residual sum of squares = 16.8199
Average test error = 0.1049
Number of variables = 11
Number of significant variables = 8
Proportion of significant variables = 73%

BIC-selected model:
AUC = 0.928
Average residual sum of squares = 19.8541
Average test error = 0.1083
Number of variables = 9
Number of significant variables = 9
Proportion of significant variables = 100%

As these results show, the AIC model has the best overall fit: it achieves the smallest fitting error on the full dataset and the smallest average test error in cross-validation, and it clearly outperforms the full model. In terms of parsimony and “cleanliness” of the predictors, it is slightly inferior to the BIC model, which retains fewer variables and makes all of them significant.

By contrast, the lasso model under the optimal tuning parameter retained only 6 variables, but its recall was only 54.36%, making it insufficiently reliable for this analysis. Therefore, it was not adopted.

Taking all factors into account (including interpretability), this study ultimately adopts the AIC-selected model for subsequent modeling.
