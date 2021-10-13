# This file seeks to test the performance of the algorithms by applying the results to a selfmade game.
# The game will have the following rules:
#     -The initial stake will be at 100.000 of currency
#     -The full position is bought/sold/hold for each position.
#     -The brokerage for each position, taken will be statically set to 0.06% of the entire position. This value is lost to the broker.
#     -The game will be based on a loop through the columns y_pred and y_test. As we work with 3 classes the possible
#      outcome must be 3**2= 9 cases. The game will reward/penaltise the position as follows in each cases:
#     The game rules will be based on 9 individual cases described below: {
#         Case (True 1, predicted 1):
#             -The position will grow as: ((Position – brokerage)*1.03)-brokerage
#         Case (True 1, predicted 2):
#             -The position will grow as: No position taken
#         Case (True 1, predicted 3):
#             -The position will grow as: ((Position – brokerage)*-1.03)-brokerage
#         Case (True 3, predicted 1):
#             -The position will grow as: ((Position – brokerage)*(-1.03))-brokerage
#         Case (True 3, predicted 2):
#             - The position will grow as: No position taken
#         Case (True 3, predicted 3):
#             -The position will grow as: ((Position – brokerage)*(1.03))-brokerage
#         Case (True 2, predicted 1):
#             -The position will grow as: ((Position – brokerage)*rand(-1.5,1.5)-brokerage
#         Case (True 2, predicted 2):
#             -The position will grow as: No position taken
#         Case (True 2, predicted 3):
#             -The position will grow as: ((Position – brokerage)*rand(-1.5,1.5)-brokerage




