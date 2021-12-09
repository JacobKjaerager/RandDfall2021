import random
import matplotlib.pyplot as plt
def update_initail_stake(old_price_minus_brokage,
                         fortune_development,
                         brokerage):
    saleable_price = old_price_minus_brokage*(1+fortune_development)
    return saleable_price*(1 - 1*brokerage)

def run_game(df,
             initial_stake: int = 100,
             brokerage: float = 0.0004,
             correct_return: float = 0.002,
             wrong_return: float = -0.002,
             uniform_parameter: float= 0.002,
             legend_name: str = "test"):

    current_fortune = []
    possible_positions = []
    for index, row in df.iterrows():
        possible_positions.append(index)
        current_fortune.append(initial_stake)
        price_after_purchase_brokage = initial_stake * (1 - brokerage)
        if (row["real"] == 1 and row["predicted"] == 1):
            initial_stake = update_initail_stake(old_price_minus_brokage=price_after_purchase_brokage,
                                                 fortune_development=correct_return ,
                                                 brokerage=brokerage)

        elif (row["real"] == 1 and row["predicted"] == 3):
            initail_stake = update_initail_stake(old_price_minus_brokage=price_after_purchase_brokage,
                                                 fortune_development=wrong_return,
                                                 brokerage=brokerage)

        elif (row["real"] == 2 and row["predicted"] == 1):
            initial_stake = update_initail_stake(old_price_minus_brokage=price_after_purchase_brokage,
                                                 fortune_development=random.uniform(-uniform_parameter, uniform_parameter),
                                                 brokerage=brokerage)
        elif (row["real"] == 2 and row["predicted"] == 3):
            initial_stake = update_initail_stake(old_price_minus_brokage=price_after_purchase_brokage,
                                                 fortune_development=random.uniform(-uniform_parameter, uniform_parameter),
                                                 brokerage=brokerage)
        elif (row["real"] == 3 and row["predicted"] == 1):
            initail_stake = update_initail_stake(old_price_minus_brokage=price_after_purchase_brokage,
                                                 fortune_development=wrong_return,
                                                 brokerage=brokerage)

        elif (row["real"] == 3 and row["predicted"] == 3):
            initial_stake = update_initail_stake(old_price_minus_brokage=price_after_purchase_brokage,
                                                 fortune_development=correct_return,
                                                 brokerage=brokerage)

    plt.plot(possible_positions, current_fortune, label='{}'.format(legend_name))
    plt.legend(loc="upper left")
    plt.title("End_fortune: {}, brokerage: {}, correct_return: {}, wrong_return: {}, uniform_parameter: {}".format(
        initial_stake, brokerage, correct_return, wrong_return,uniform_parameter
    ))
    plt.show()
    return initial_stake

if __name__ == '__main__':
    df2 = pd.read_csv("../global_saves/ensemble_pred_ceil_2_floor_0_002_deduction_factor.csv").drop(colums=[[]])
    for i in df2.columns:
        if i != "real":
            df = df2[["real", i]].rename(columns={i: "predicted"})
            run_game(df, brokerage=0.00089, legend_name=i)