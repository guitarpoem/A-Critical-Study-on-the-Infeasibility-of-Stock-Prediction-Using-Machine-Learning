# The Limits of Learning: A Critical Study on the Infeasibility of Stock Prediction Using Machine Learning

## Experimental Results

### Classification Performance

### Regression Performance

### Persistence Model Comparison

### Infeasibility of Predicting Stock Movements Direction

As shown in Table 1, the classification task of predicting stock movements direction achieves accuracy equal to or lower than simply predicting the dominant class in the test set. This pattern suggests strong overfitting and indicates that the models fail to learn meaningful patterns beyond the class distribution in the training data.

### Infeasibility of Predicting Stock Prices

Table 2 shows R² values for price-based models ranging from 0.6230 to 0.9514 across six stocks. While these values appear promising, they do not indicate true predictive power.

Table 3 reveals a critical insight: despite the LSTM models' seemingly good R² values (ranging from 0.6230 to 0.9514), they consistently underperform compared to the naive persistence model. This pattern persists across all six stocks, with the persistence model achieving higher R² values in every case. The LSTM's apparent success is thus revealed to be an illusion - rather than learning meaningful patterns, the models simply default to predicting values close to the previous day's price, mirroring the persistence model's behavior.

This phenomenon aligns with findings by Leccese [12], who demonstrated LSTM's same behavior in predicting stock prices. Leccese's analysis of U.S. market data from 1950-2018 showed an R² of 0.6976 for such models, with even higher values possible over shorter time periods. This behavior is precisely what would be expected from a model with no actual predictive ability beyond the strong autocorrelation inherent in price time series.

The models' failure to learn meaningful patterns is further evidenced by the overfitting observed in the classification task (Table 1). Notably, both the classification and regression tasks employ identical data splits and model architectures, differing only in their output layers. Given that predicting exact price values is inherently more complex than forecasting directional movements, the regression task should present greater difficulty. Consequently, the apparently strong regression performance using the same model configuration as the poorly-performing classification task must be viewed as illusory, reflecting the models' tendency to default to persistence predictions rather than genuine learning.

### Sentiment Label Effectiveness

Moreover, the sentiment generated from social media data doesn't provide meaningful signal to improve the results, as shown in Table 2. For five of the six stocks (AAPL, AMZN, BAC, D, and GOOG), adding sentiment data resulted in higher MAE and lower R² values. Only Citigroup (C) showed marginal improvement with sentiment data, and even this improvement was negligible. The degradation is particularly significant for Amazon (AMZN), where MAE more than doubled from 10.21 to 23.24 when sentiment was incorporated, while R² dropped substantially from 0.9514 to 0.7908. This pattern strongly suggests that social media sentiment introduces noise rather than signal into the prediction process.

To investigate the quality of the LLM generated sentiment labels, we examined the correlation between today's LLM-generated sentiment and the direction of next-day stock movements. Table 4 shows the correlation strength measured using Cramer's V coefficient. The analysis reveals consistently weak and statistically insignificant correlations between sentiment and price movements (all V < 0.1, p > 0.05). Prediction accuracies (29.72%-34.99%) were only slightly better than random chance, indicating social media sentiment offers minimal predictive value for stock movements.

We further examine Amazon as a case study. Table 5 shows the distribution of sentiment labels and price movements. Chi-square analysis revealed no statistically significant relationship between sentiment and subsequent price movements (p-value = 0.8572). The nearly uniform distribution across all cells in Table 6 visually confirms this lack of predictive relationship - each sentiment category shows almost identical distributions across all three price movement outcomes. This result demonstrates that even sophisticated LLM-generated sentiment labels fail to capture any meaningful signal for predicting next-day stock movements, further supporting the efficient market hypothesis.

### Sentiment Correlation Analysis

### AMZN Case Study

## Discussion: What ML Can Do --- and What It Can't

## References

[12] Leccese, F. (2019). Deep Learning for Stock Market Prediction: A Critical Analysis. Journal of Financial Data Science, 1(2), 45-62. 