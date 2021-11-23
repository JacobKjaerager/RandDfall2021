import random

def update_initail_stake(price_after_purchase_brokage, return_type, brokerage):
    saleable_price = price_after_purchase_brokage*(1+return_type)
    return saleable_price*(1 - 1*brokerage)

def run_game(df,
             initial_stake: int = 100,
             brokerage: float = 0.0004,
             correct_return: float = 0.002,
             wrong_return: float = -0.002,
             uniform_parameter: float= 0.002):
    current_fortune = []
    possible_positions = []
    for index, row in df.head(2000).iterrows():
        possible_positions.append(index)
        current_fortune.append(initial_stake)
        price_after_purchase_brokage = initial_stake * (1 - brokerage)
        if (row["real"] == 1 and row["predicted"] == 1):
            initial_stake = update_initail_stake(price_after_purchase_brokage, correct_return, brokerage)

        elif (row["real"] == 1 and row["predicted"] == 3):
            initail_stake = update_initail_stake(price_after_purchase_brokage, wrong_return, brokerage)

        elif (row["real"] == 2 and row["predicted"] == 1):
            initial_stake = update_initail_stake(price_after_purchase_brokage,
                                                 random.uniform(-uniform_parameter, uniform_parameter), brokerage)
        elif (row["real"] == 2 and row["predicted"] == 3):
            initial_stake = update_initail_stake(price_after_purchase_brokage,
                                                 random.uniform(-uniform_parameter, uniform_parameter), brokerage)
        elif (row["real"] == 3 and row["predicted"] == 1):
            initail_stake = update_initail_stake(price_after_purchase_brokage, wrong_return, brokerage)

        elif (row["real"] == 3 and row["predicted"] == 3):
            initial_stake = update_initail_stake(price_after_purchase_brokage, correct_return, brokerage)

    plt.plot(possible_positions, current_fortune)
    plt.show()
    return initial_stake

