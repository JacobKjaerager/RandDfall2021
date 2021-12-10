import random
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.metrics import accuracy_score

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

    return [possible_positions, current_fortune, legend_name]


def save_weights(df):
    data = []
    for i in df.columns:
        data.append(
            go.Scatter(
                x=df.index,
                y=df[i],
                name="{}".format(i)
            )
        )
    fig = go.Figure(
        data=data
    )
    fig.update_layout(title="Weight development over time",
                      xaxis_title='Possible positions',
                      yaxis_title="Current weight")
    fig.write_html("{}_{}.html".format("../global_saves/", "weights_for_ensemble"))


def save_ensemble_and_stuff(df):
    q = []
    data = []
    brokerage= 0.00089
    initial_stake= 100
    correct_return=0.002
    wrong_return=-0.002
    uniform_parameter=0.002
    for i in df.columns:
        if i != "real":
            df2 = df[["real", i]].rename(columns={i: "predicted"})
            q.append(run_game(df=df2,
                              brokerage= brokerage,
                              initial_stake= initial_stake ,
                              correct_return=correct_return,
                              wrong_return=wrong_return,
                              uniform_parameter=uniform_parameter,
                              legend_name=i)
             )
    for i in q:
        data.append(
            go.Scatter(
                x=i[0],
                y=i[1],
                name="{}".format(i[2] + ", Acc: {}".format(round(accuracy_score(df["real"], df[i[2]]),3)))
            )
        )
    fig = go.Figure(
        data=data
    )
    fig.update_layout(title="End_fortune: {}, brokerage: {}, correct_return: {}, wrong_return: {}, uniform_parameter: {}".format(
        initial_stake, brokerage, correct_return, wrong_return, uniform_parameter
    ),
                      xaxis_title='Possible position',
                      yaxis_title="Current fortune")
    fig.write_html("{}_{}.html".format("../global_saves/", "ensemble_pred_comp"))

if __name__ == '__main__':
    df1 = pd.read_csv("../global_saves/ensemble_pred_ceil_2_floor_0_002_deduction_factor.csv").drop(colums=[[]])


