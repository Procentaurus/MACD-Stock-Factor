import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D


def count_exp(alfa, exp):
    return (1-alfa)**exp


def count_ema(numbers, n):  # function counts EMA index (n periods)
    alfa = 2/(n+1)
    ema_table = []

    for i in range(len(numbers)):

        p_table = []
        for j in reversed(range(i - n + 1, i+1)):  # taking values of needed days
            p_table.append(numbers[j]) if j >= 0 else p_table.append(-1)

        numerator = 0
        denominator = 0
        for p in range(len(p_table)):  # calculating numerator and denumenator of EMA fraction
            if p_table[p] != -1:
                value1 = p_table[p] * count_exp(alfa, p)
                numerator += value1

                value2 = count_exp(alfa, p)
                denominator += value2
            else:
                break

        ema_table.append(numerator / denominator)
    return ema_table


def find_intersections(data1, data2):  # function looks for coordinates and their types between MACD chart and Signal chart
    x_coordinates_of_itersections = []
    y_coordinates_of_itersections = []
    types_of_itersections = []

    for i in range(1, len(data1)):
        if data1[i-1] >= data2[i-1] and data1[i] <= data2[i]:
            types_of_itersections.append("sell")
            x_coordinates_of_itersections.append(i)
            y_coordinates_of_itersections.append((data1[i - 1] + data1[i]) / 2)  # values are approximated
        elif data1[i-1] <= data2[i-1] and data1[i] >= data2[i]:
            types_of_itersections.append("buy")
            x_coordinates_of_itersections.append(i)
            y_coordinates_of_itersections.append((data1[i - 1] + data1[i]) / 2) # values are approximated

    return x_coordinates_of_itersections, y_coordinates_of_itersections, types_of_itersections


def print_chart_1(data1, data2, x_intersections=None, y_intersections=None):  # function to plot the chart with MACD and Signal
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3, data4 = None, None

    if x_intersections is not None and y_intersections is not None:
        data3 = np.array(x_intersections)
        data4 = np.array(y_intersections)

    plt.figure(figsize=(16, 6))

    axis = plt.gca()
    axis.set_xlim([0, 1000])

    plt.plot(data1, label='MACD', zorder=10)
    plt.plot(data2, label='Signal', zorder=5)
    if data3 is not None and data4 is not None:
        plt.scatter(data3, data4, marker='+', color='black', zorder=15)


    plt.title(label="Wykres wskaźników MACD i Signal", fontsize=30, pad=20)
    plt.legend(loc="upper right")

    plt.show()


def print_chart_2(data1, data2, x_intersections=None, types_intersections=None):  # function to plot the chart of exchange rates of
                                                                                  # pound sterling
    data1 = np.array(data1)
    data2 = np.array(data2)
    data3, data4 = None, None

    if x_intersections is not None and types_intersections is not None:
        data3 = np.array(x_intersections)
        data4 = np.array(types_intersections)

    data1 = pd.to_datetime(data1)

    plt.figure(figsize=(16, 6))

    axis = plt.gca()
    axis.set_xlim([datetime.date(2019, 3, 25), datetime.date(2023, 3, 9)])
    axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f zł'))
    plt.title(label="Wykres kursu funta szterlinga do złotówki", fontsize=30, pad=20)

    plt.plot(data1, data2, zorder=10)
    if data3 is not None and data4 is not None:  # optional drawing points of buy and sell

        data5 = []
        for text in data4:
            data5.append("green") if text == "buy" else data5.append("red")

        data6 = return_rates_of_actions(data2, data3)
        data7 = return_dates_of_actions(data1, data3)

        data6 = np.array(data6)
        data5 = np.array(data5)
        plt.scatter(data7, data6, marker='o', c=data5, zorder=15)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Kupno', markerfacecolor='g', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Sprzedaż', markerfacecolor='r', markersize=10)]
        plt.legend(handles=legend_elements)
    plt.show()


def return_dates_of_actions(all_dates, indexes_of_actions):
    dates_of_actions = []
    for i in range(len(indexes_of_actions)):
        dates_of_actions.append(all_dates[indexes_of_actions[i]])

    return dates_of_actions


def return_rates_of_actions(all_rates, indexes_of_actions):  # function returns all values of arguments for which there has been
    rates_of_actions = []                                    # intersection of MACD and Signal
    for i in range(len(indexes_of_actions)):  # reversing values due to fact that x axis consists of dates
        indexes_of_actions[i] = 1000 - indexes_of_actions[i]
        rates_of_actions.append(all_rates[int(indexes_of_actions[i])])

    return rates_of_actions


def makeSimulation(start_quantity, prices, actions, commision=0.0):  # function simulates activity of investor who buys and sells
    accountPLN = start_quantity                                      # at the moments chosen by MACD and Signal
    accountGBP = 0

    for i in range(len(actions)):
        if actions[i] == "buy" and accountPLN != 0 and i != len(actions) - 1:
            if commision == 0:
                accountGBP = accountPLN/prices[i]
            else:
                accountGBP = accountPLN/prices[i] * (1 - commision)
            accountPLN = 0
        elif actions[i] == "sell" and accountGBP != 0:
            if commision == 0:
                accountPLN = accountGBP * prices[i]
            else:
                accountPLN = accountGBP * prices[i] * (1 - commision)
            accountGBP = 0

    return accountPLN

def print_simulation_results():

    rates_of_actions = return_rates_of_actions(exchange_rates, x_coordinates_of_itersections)

    resPLN = makeSimulation(1000, rates_of_actions, types_of_itersections)
    print("Simulation end result is : " + str(round(resPLN, 2)) + " zł (without commision for broker.)")
    resPLN2 = makeSimulation(1000, rates_of_actions, types_of_itersections, 0.001)
    print("Simulation end result is : " + str(round(resPLN2, 2)) + " zł (with 0.1% commision for broker.)")


def calculate_macd_and_signal(rates):
    ema12 = count_ema(rates, 12)  # calculate needed values of EMAn, MACD nad Signal
    ema26 = count_ema(rates, 26)

    macd = []
    for i in range(len(ema12)):
        macd.append(ema12[i] - ema26[i])

    signal = count_ema(macd, 9)

    return macd, signal


def readData():
    data = pd.read_csv('D:/UCZELNIA/sem4/Metody Numeryczne/Project_1_MACD/moneypl.csv')  # read data
    temp1 = list(data["Data"][0:1000])
    temp2 = list(data["Kurs średni"][0:1000])

    return temp1, temp2


dates, exchange_rates = readData()
macd, signal = calculate_macd_and_signal(exchange_rates)
x_coordinates_of_itersections, y_coordinates_of_itersections, types_of_itersections = find_intersections(signal, macd)

print_chart_1(macd, signal)
print_chart_1(macd, signal, x_coordinates_of_itersections, y_coordinates_of_itersections)
print_chart_2(dates, exchange_rates)
print_chart_2(dates, exchange_rates, x_coordinates_of_itersections, types_of_itersections)
print_simulation_results()
