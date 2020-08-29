The following iterations were done on the previous architecture of Session-4
1. L1 + BN
2. L2 + BN
3. L1 and L2 with BN
4. GBN
5.L1 and L2 with GBN

Observations:
1. GBN is found to be performing well with less validation loss and high validation accuracy
2. L1 regularization was not performing well as compared to L2 and GBN as it is eliminating parameters by making them zero
3. After GBN, L2 regularization is found to be performing well

