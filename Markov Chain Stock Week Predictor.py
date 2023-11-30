import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import yfinance as yf
from scipy.stats import shapiro
import statistics
import random
from datetime import datetime, timedelta
import ta
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def fetch_data_list(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Close'].tolist()
    return prices

start_date = datetime(2011,3,1)
end_date = datetime(2023,1,1)
current_date = start_date
beginning_date = start_date + timedelta(days = 1200)

while beginning_date < end_date:
    current_date += timedelta(days = 1)
    beginning_date += timedelta(days = 1)
    if beginning_date.weekday() < 5:
        apple_list = fetch_data_list('SPY', current_date, beginning_date)
        qqq_list = fetch_data_list('QQQ', current_date, beginning_date)

        for i in range(3, len(apple_list)):
            apple_list[i] = round(apple_list[i], 2)

        for i in range(3,len(apple_list)):
            qqq_list[i] = round(qqq_list[i],2)

        apple_fifty_moving = [0]*len(apple_list)
        apple_twohundred_moving = [0]*len(apple_list)

        for i in range(55, len(apple_list)):
            apple_fifty_moving[i] = statistics.mean(apple_list[i-55:i-5])

        for i in range(205, len(apple_list)):
            apple_twohundred_moving[i] = statistics.mean(apple_list[i-205:i-5])

        apple_difference_moving = [0]*len(apple_list)
        for i in range(205, len(apple_list)):
            apple_difference_moving[i] = 100*(apple_fifty_moving[i] - apple_twohundred_moving[i])/apple_twohundred_moving[i]

        apple_day_change = [0]*len(apple_list)
        for i in range(2,len(apple_list)):
            apple_day_change[i] = (apple_list[i] - apple_list[i-1])/apple_list[i-1]
            apple_day_change[i] = round(apple_day_change[i],2)
        applepastweek_percent_change = [0]*len(apple_list)
        for i in range(10, len(apple_list)):
            applepastweek_percent_change[i] = 100*(apple_list[i-5] - apple_list[i-9])/(apple_list[i-9])

        qqqpastweek_percent_change = [0]*len(apple_list)
        for i in range(5, len(apple_list)):
            qqqpastweek_percent_change[i] = 100*(qqq_list[i-5] - qqq_list[i-9])/(qqq_list[i-9])

        applenextweek_percent_change = [0]*len(apple_list)
        for i in range(5, len(apple_list)):
            applenextweek_percent_change[i] = 100*(apple_list[i] - apple_list[i-4])/(apple_list[i-4])

        week_percent_ranges = [-100,-6,-4,-3,-2,-1,-.5,0,.5,1,2,3,4,6,100]

        data = yf.download('SPY', current_date, beginning_date)
        macd = ta.trend.macd_diff(data['Close'])
        signal_line = ta.trend.macd_signal(data['Close'])
        macd_list = macd.dropna().tolist()
        signal_list = signal_line.dropna().tolist()

        def calculate_obv(data):
            obv = []
            prev_obv = 0

            for i in range(1, len(data)):
                if data['Close'][i] > data['Close'][i - 1]:
                    obv.append(prev_obv + data['Volume'][i])
                else:
                    obv.append(prev_obv)

                prev_obv = obv[-1]

            return pd.Series(obv, index=data.index[1:])

        obv = calculate_obv(data)

        # Plot the results
        obv_list = obv.dropna().tolist()
        average_obv = [0]*len(obv_list)

        for i in range(15,len(obv_list)):
            average_obv[i] = statistics.mean(obv_list[i-14:i-5])

        current_obv = [0]*len(obv_list)

        for i in range(6,len(obv_list)):
            current_obv[i] = obv_list[i-5]

        current_obv = [0] + current_obv
        average_obv = [0] + average_obv
        obv_difference = [0]* len(current_obv)

        for i in range(16,len(current_obv)):
            obv_difference[i] = 100*(current_obv[i] - average_obv[i])/current_obv[i]

        macd_list = [0]*33 + macd_list
        signal_list = [0]*33 + signal_list
        new_macd_list = [0]*len(macd_list)
        new_signal_list = [0]*len(macd_list)

        for i in range(35,len(macd_list)):
            new_macd_list[i] = macd_list[i-5]
            new_macd_list[i] = round(macd_list[i], 2)

        for i in range(35,len(macd_list)):
            new_signal_list[i] = signal_list[i-5]
            new_signal_list[i] = round(signal_list[i], 2)

        macd_list = new_macd_list
        signal_list = new_signal_list

        difference = [0]*len(macd_list)
        for i in range(35, len(macd_list)):
            difference[i] = macd_list[i] - signal_list[i]
            difference[i] = round(difference[i], 2)

        rows = 100
        cols = 14
        matrix_apple_qqq = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if qqqpastweek_percent_change[i] < variable_percent_ranges[j] and qqqpastweek_percent_change[i] > variable_percent_ranges[j-1] and applepastweek_percent_change[i] < variable_percent_ranges[a] and applepastweek_percent_change[i] > variable_percent_ranges[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_apple_qqq[row_counter][col_counter] = counter

        rows = 100
        cols = 14
        matrix_apple_macd = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if difference[i] < difference_range[j] and difference[i] > difference_range[j-1] and applepastweek_percent_change[i] < variable_percent_ranges[a] and applepastweek_percent_change[i] > variable_percent_ranges[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_apple_macd[row_counter][col_counter] = counter

        rows = 100
        cols = 14
        matrix_apple_obv = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if obv_difference[i] < obv_range[j] and obv_difference[i] > obv_range[j-1] and applepastweek_percent_change[i] < variable_percent_ranges[a] and applepastweek_percent_change[i] > variable_percent_ranges[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_apple_obv[row_counter][col_counter] = counter

        rows = 100
        cols = 14
        matrix_apple_moving = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if apple_difference_moving[i] < moving_range[j] and apple_difference_moving[i] > moving_range[j-1] and applepastweek_percent_change[i] < variable_percent_ranges[a] and applepastweek_percent_change[i] > variable_percent_ranges[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_apple_moving[row_counter][col_counter] = counter

        rows = 100
        cols = 14
        matrix_macd_obv = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if obv_difference[i] < obv_range[j] and obv_difference[i] > obv_range[j-1] and difference[i] < difference_range[a] and difference[i] > difference_range[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_macd_obv[row_counter][col_counter] = counter

        rows = 100
        cols = 14
        matrix_macd_moving = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if apple_difference_moving[i] < moving_range[j] and apple_difference_moving[i] > moving_range[j-1] and difference[i] < difference_range[a] and difference[i] > difference_range[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_macd_moving[row_counter][col_counter] = counter

        rows = 100
        cols = 14
        matrix_obv_moving = np.zeros((rows,cols))

        counter = 0
        row_counter = -1
        col_counter = -1

        variable_percent_ranges = [-100,-4,-3,-2,-1,0,1,2,3,4,100]
        difference_range = [-100,-4,-2,-1,-.5,0,.5,1,2,4,100]
        obv_range = [-100,0,.2,.3,.4,.5,.75,1,1.5,2,100]
        moving_range = [-100,-10,-5,-2,0,2,6,8,10,15,100]

        for a in range(1, len(variable_percent_ranges)):
            for j in range(1, len(variable_percent_ranges)):
                col_counter = -1
                row_counter += 1
                for k in range(1, len(week_percent_ranges)):
                    col_counter += 1
                    counter = 0
                    for i in range(1, len(apple_list)):
                        if apple_difference_moving[i] < moving_range[j] and apple_difference_moving[i] > moving_range[j-1] and obv_difference[i] < obv_range[a] and obv_difference[i] > obv_range[a-1] and applenextweek_percent_change[i] < week_percent_ranges[k] and applenextweek_percent_change[i] > week_percent_ranges[k-1]:
                            counter += 1
                        matrix_obv_moving[row_counter][col_counter] = counter
        
        current_apple_counter = 0
        current_qqq_counter = 0
        current_macd_counter = 0
        current_obv_counter = 0
        current_moving_counter = 0

        current_applepastweek_percent_change = applepastweek_percent_change[len(applepastweek_percent_change) - 1]
        current_qqqpastweek_percent_change = qqqpastweek_percent_change[len(qqqpastweek_percent_change) - 1]
        current_difference = difference[len(applepastweek_percent_change) - 1]
        current_obv = obv_difference[len(applepastweek_percent_change) - 1]
        current_moving = apple_difference_moving[len(applepastweek_percent_change) - 1]

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_applepastweek_percent_change < variable_percent_ranges[i] and current_applepastweek_percent_change > variable_percent_ranges[i-1]:
                current_apple_counter = counter
            else:
                pass    

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_qqqpastweek_percent_change < variable_percent_ranges[i] and current_qqqpastweek_percent_change > variable_percent_ranges[i-1]:
                current_qqq_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_difference < difference_range[i] and current_difference > difference_range[i-1]:
                current_difference_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_obv < obv_range[i] and current_obv > obv_range[i-1]:
                current_obv_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_moving < moving_range[i] and current_moving > moving_range[i-1]:
                current_moving_counter = counter
            else:
                pass  

        apple_qqq_list = [0]*14
        apple_macd_list = [0]*14
        apple_obv_list = [0]*14
        apple_moving_list = [0]*14
        macd_obv_list = [0]*14
        macd_moving_list = [0]*14
        obv_moving_list = [0]*14

        apple_qqq_list = matrix_apple_qqq[10*current_apple_counter+current_qqq_counter]
        apple_macd_list = matrix_apple_macd[10*current_apple_counter+current_macd_counter]
        apple_obv_list = matrix_apple_obv[10*current_apple_counter+current_obv_counter]
        apple_moving_list = matrix_apple_moving[10*current_apple_counter+current_moving_counter]
        macd_obv_list = matrix_macd_obv[10*current_macd_counter+current_obv_counter]
        macd_moving_list = matrix_macd_moving[10*current_macd_counter+current_moving_counter]
        obv_moving_list = matrix_macd_moving[10*current_obv_counter+current_moving_counter]

        week_percent_ranges = [-100,-6,-4,-3,-2,-1,-.5,0,.5,1,2,3,4,6,100]

        expected_value_ranges = [-6,-4,-3,-2,-1,-.5,-.25,.25,.5,1,2,3,4,6]
        
        apple_qqq_ev = 0
        apple_macd_ev = 0
        apple_obv_ev = 0
        apple_moving_ev = 0
        macd_obv_ev = 0
        macd_moving_ev = 0
        obv_moving_ev = 0

        for i in range(0,14):
            apple_qqq_ev += apple_qqq_list[i]*expected_value_ranges[i]
            apple_macd_ev += apple_macd_list[i]*expected_value_ranges[i]
            apple_obv_ev += apple_obv_list[i]*expected_value_ranges[i]
            apple_moving_ev += apple_moving_list[i]*expected_value_ranges[i]
            macd_obv_ev += macd_obv_list[i]*expected_value_ranges[i]
            macd_moving_ev += macd_moving_list[i]*expected_value_ranges[i]
            obv_moving_ev += obv_moving_list[i]*expected_value_ranges[i]

        sum_apple_qqq = 0
        sum_apple_macd = 0 
        sum_apple_obv = 0
        sum_apple_moving = 0
        sum_macd_obv = 0
        sum_macd_moving = 0
        sum_obv_moving = 0

        total = 0

        for i in range(7,14):
            sum_apple_qqq += apple_qqq_list[i]

        for i in range(7,14):
            sum_apple_macd += apple_macd_list[i]

        for i in range(7,14):
            sum_apple_obv += apple_obv_list[i]

        for i in range(7,14):
            sum_apple_moving += apple_moving_list[i]
        
        for i in range(7,14):
            sum_macd_obv += macd_obv_list[i]
        
        for i in range(7,14):
            sum_macd_moving += macd_moving_list[i]

        for i in range(7,14):
            sum_obv_moving += obv_moving_list[i]

        for i in range(0,14):
            total += apple_qqq_list[i]
            total += apple_macd_list[i]
            total += 2*apple_obv_list[i]
            total += 2*apple_moving_list[i]
            total += macd_obv_list[i]
            total += macd_moving_list[i]
            total += obv_moving_list[i]

        prime1 = sum_apple_qqq
        prime2 = sum_apple_macd
        prime3 = sum_apple_obv
        prime4 = sum_apple_moving
        prime5 = sum_macd_moving
        prime6 = sum_macd_obv
        prime7 = sum_obv_moving
        
        total_expected_value = apple_qqq_ev+apple_macd_ev+2*apple_obv_ev+2*apple_moving_ev+macd_obv_ev+macd_moving_ev+obv_moving_ev
        total_expected_value = round(total_expected_value, 2)
        
        total_prime_sum = sum_apple_qqq+sum_apple_macd+2*sum_apple_obv+2*sum_apple_moving+sum_macd_moving+sum_macd_obv+sum_obv_moving

        prime_percentage_profit = 100*(total_prime_sum/total)

        prime_percentage_profit = round(prime_percentage_profit,2)
    


        current_apple_counter = 0
        current_qqq_counter = 0
        current_macd_counter = 0
        current_obv_counter = 0
        current_moving_counter = 0

        current_applepastweek_percent_change = applepastweek_percent_change[len(applepastweek_percent_change) - 1]
        current_qqqpastweek_percent_change = qqqpastweek_percent_change[len(qqqpastweek_percent_change) - 1]
        current_difference = difference[len(applepastweek_percent_change) - 1]
        current_obv = obv_difference[len(applepastweek_percent_change) - 1]
        current_moving = apple_difference_moving[len(applepastweek_percent_change) - 1]

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_applepastweek_percent_change < variable_percent_ranges[i] and current_applepastweek_percent_change > variable_percent_ranges[i-1]:
                current_apple_counter = counter
            else:
                pass    

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_qqqpastweek_percent_change < variable_percent_ranges[i] and current_qqqpastweek_percent_change > variable_percent_ranges[i-1]:
                current_qqq_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_difference < difference_range[i] and current_difference > difference_range[i-1]:
                current_difference_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_obv < obv_range[i] and current_obv > obv_range[i-1]:
                current_obv_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_moving < moving_range[i] and current_moving > moving_range[i-1]:
                current_moving_counter = counter
            else:
                pass  

        apple_qqq_list = [0]*14
        apple_macd_list = [0]*14
        apple_obv_list = [0]*14
        apple_moving_list = [0]*14
        macd_obv_list = [0]*14
        macd_moving_list = [0]*14
        obv_moving_list = [0]*14

        if current_apple_counter > 0:
            current_apple_counter -= 1
        
        if current_macd_counter > 0:
            current_macd_counter -= 1
        
        if current_obv_counter > 0:
            current_obv_counter -= 1

        if current_moving_counter > 0:
            current_moving_counter -= 1 

        apple_qqq_list = matrix_apple_qqq[10*current_apple_counter+current_qqq_counter]
        apple_macd_list = matrix_apple_macd[10*current_apple_counter+current_macd_counter]
        apple_obv_list = matrix_apple_obv[10*current_apple_counter+current_obv_counter]
        apple_moving_list = matrix_apple_moving[10*current_apple_counter+current_moving_counter]
        macd_obv_list = matrix_macd_obv[10*current_macd_counter+current_obv_counter]
        macd_moving_list = matrix_macd_moving[10*current_macd_counter+current_moving_counter]
        obv_moving_list = matrix_macd_moving[10*current_obv_counter+current_moving_counter]

        sum_apple_qqq = 0
        sum_apple_macd = 0 
        sum_apple_obv = 0
        sum_apple_moving = 0
        sum_macd_obv = 0
        sum_macd_moving = 0
        sum_obv_moving = 0

        apple_qqq_ev = 0
        apple_macd_ev = 0
        apple_obv_ev = 0
        apple_moving_ev = 0
        macd_obv_ev = 0
        macd_moving_ev = 0
        obv_moving_ev = 0

        for i in range(0,14):
            apple_qqq_ev += apple_qqq_list[i]*expected_value_ranges[i]
            apple_macd_ev += apple_macd_list[i]*expected_value_ranges[i]
            apple_obv_ev += apple_obv_list[i]*expected_value_ranges[i]
            apple_moving_ev += apple_moving_list[i]*expected_value_ranges[i]
            macd_obv_ev += macd_obv_list[i]*expected_value_ranges[i]
            macd_moving_ev += macd_moving_list[i]*expected_value_ranges[i]
            obv_moving_ev += obv_moving_list[i]*expected_value_ranges[i]

        total = 0

        for i in range(7,14):
            sum_apple_qqq += apple_qqq_list[i]

        for i in range(7,14):
            sum_apple_macd += apple_macd_list[i]

        for i in range(7,14):
            sum_apple_obv += apple_obv_list[i]

        for i in range(7,14):
            sum_apple_moving += apple_moving_list[i]
        
        for i in range(7,14):
            sum_macd_obv += macd_obv_list[i]
        
        for i in range(7,14):
            sum_macd_moving += macd_moving_list[i]

        for i in range(7,14):
            sum_obv_moving += obv_moving_list[i]

        for i in range(0,14):
            total += apple_qqq_list[i]
            total += apple_macd_list[i]
            total += 2*apple_obv_list[i]
            total += 2*apple_moving_list[i]
            total += macd_obv_list[i]
            total += macd_moving_list[i]
            total += obv_moving_list[i]

        underprime1 = sum_apple_qqq
        underprime2 = sum_apple_macd
        underprime3 = sum_apple_obv
        underprime4 = sum_apple_moving
        underprime5 = sum_macd_moving
        underprime6 = sum_macd_obv
        underprime7 = sum_obv_moving

        total_underexpected_value = apple_qqq_ev+apple_macd_ev+2*apple_obv_ev+2*apple_moving_ev+macd_obv_ev+macd_moving_ev+obv_moving_ev
        total_underexpected_value = round(total_underexpected_value, 2)

        total_underprime_sum = sum_apple_qqq+sum_apple_macd+2*sum_apple_obv+2*sum_apple_moving+sum_macd_moving+sum_macd_obv+sum_obv_moving

        under_percentage_profit = 100*(total_underprime_sum/total)

        under_percentage_profit = round(under_percentage_profit,2)
    
        current_apple_counter = 0
        current_qqq_counter = 0
        current_macd_counter = 0
        current_obv_counter = 0
        current_moving_counter = 0

        current_applepastweek_percent_change = applepastweek_percent_change[len(applepastweek_percent_change) - 1]
        current_qqqpastweek_percent_change = qqqpastweek_percent_change[len(qqqpastweek_percent_change) - 1]
        current_difference = difference[len(applepastweek_percent_change) - 1]
        current_obv = obv_difference[len(applepastweek_percent_change) - 1]
        current_moving = apple_difference_moving[len(applepastweek_percent_change) - 1]

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_applepastweek_percent_change < variable_percent_ranges[i] and current_applepastweek_percent_change > variable_percent_ranges[i-1]:
                current_apple_counter = counter
            else:
                pass    

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_qqqpastweek_percent_change < variable_percent_ranges[i] and current_qqqpastweek_percent_change > variable_percent_ranges[i-1]:
                current_qqq_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_difference < difference_range[i] and current_difference > difference_range[i-1]:
                current_difference_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_obv < obv_range[i] and current_obv > obv_range[i-1]:
                current_obv_counter = counter
            else:
                pass  

        counter = -1
        for i in range(1, len(difference_range)):
            counter += 1
            if current_moving < moving_range[i] and current_moving > moving_range[i-1]:
                current_moving_counter = counter
            else:
                pass  

        if current_apple_counter < 9:
            current_apple_counter += 1
        
        if current_macd_counter < 9:
            current_macd_counter += 1
        
        if current_obv_counter < 9:
            current_obv_counter += 1

        if current_moving_counter < 9:
            current_moving_counter += 1 

        apple_qqq_list = [0]*14
        apple_macd_list = [0]*14
        apple_obv_list = [0]*14
        apple_moving_list = [0]*14
        macd_obv_list = [0]*14
        macd_moving_list = [0]*14
        obv_moving_list = [0]*14

        apple_qqq_list = matrix_apple_qqq[10*current_apple_counter+current_qqq_counter]
        apple_macd_list = matrix_apple_macd[10*current_apple_counter+current_macd_counter]
        apple_obv_list = matrix_apple_obv[10*current_apple_counter+current_obv_counter]
        apple_moving_list = matrix_apple_moving[10*current_apple_counter+current_moving_counter]
        macd_obv_list = matrix_macd_obv[10*current_macd_counter+current_obv_counter]
        macd_moving_list = matrix_macd_moving[10*current_macd_counter+current_moving_counter]
        obv_moving_list = matrix_macd_moving[10*current_obv_counter+current_moving_counter]

        sum_apple_qqq = 0
        sum_apple_macd = 0 
        sum_apple_obv = 0
        sum_apple_moving = 0
        sum_macd_obv = 0
        sum_macd_moving = 0
        sum_obv_moving = 0

        apple_qqq_ev = 0
        apple_macd_ev = 0
        apple_obv_ev = 0
        apple_moving_ev = 0
        macd_obv_ev = 0
        macd_moving_ev = 0
        obv_moving_ev = 0

        for i in range(0,14):
            apple_qqq_ev += apple_qqq_list[i]*expected_value_ranges[i]
            apple_macd_ev += apple_macd_list[i]*expected_value_ranges[i]
            apple_obv_ev += apple_obv_list[i]*expected_value_ranges[i]
            apple_moving_ev += apple_moving_list[i]*expected_value_ranges[i]
            macd_obv_ev += macd_obv_list[i]*expected_value_ranges[i]
            macd_moving_ev += macd_moving_list[i]*expected_value_ranges[i]
            obv_moving_ev += obv_moving_list[i]*expected_value_ranges[i]

        total = 0

        for i in range(7,14):
            sum_apple_qqq += apple_qqq_list[i]

        for i in range(7,14):
            sum_apple_macd += apple_macd_list[i]

        for i in range(7,14):
            sum_apple_obv += apple_obv_list[i]

        for i in range(7,14):
            sum_apple_moving += apple_moving_list[i]
        
        for i in range(7,14):
            sum_macd_obv += macd_obv_list[i]
        
        for i in range(7,14):
            sum_macd_moving += macd_moving_list[i]

        for i in range(7,14):
            sum_obv_moving += obv_moving_list[i]

        for i in range(0,14):
            total += apple_qqq_list[i]
            total += apple_macd_list[i]
            total += 2*apple_obv_list[i]
            total += 2*apple_moving_list[i]
            total += macd_obv_list[i]
            total += macd_moving_list[i]
            total += obv_moving_list[i]

        overprime1 = sum_apple_qqq
        overprime2 = sum_apple_macd
        overprime3 = sum_apple_obv
        overprime4 = sum_apple_moving
        overprime5 = sum_macd_moving
        overprime6 = sum_macd_obv
        overprime7 = sum_obv_moving

        total_undexpected_value = apple_qqq_ev+apple_macd_ev+2*apple_obv_ev+2*apple_moving_ev+macd_obv_ev+macd_moving_ev+obv_moving_ev
        total_undexpected_value = round(total_undexpected_value, 2)

        final_expected_value = (total_expected_value+total_underexpected_value+total_undexpected_value)/3
        final_expected_value = round(final_expected_value,2)

        total_overprime_sum = sum_apple_qqq+sum_apple_macd+2*sum_apple_obv+2*sum_apple_moving+sum_macd_moving+sum_macd_obv+sum_obv_moving

        over_percentage_profit = 100*(total_overprime_sum/total)

        over_percentage_profit = round(over_percentage_profit,2)
    
        final_sum = (over_percentage_profit + under_percentage_profit + prime_percentage_profit)/3
        final_sum = round(final_sum, 2)


        pastweekchange = round(applepastweek_percent_change[len(applepastweek_percent_change) - 1], 2)

        nextweekchange = round(applenextweek_percent_change[len(applenextweek_percent_change) - 1 ],2)

        current_applepastweek_percent_change = round(current_applepastweek_percent_change,2)
        current_difference = round(current_difference,2)
        current_moving = round(current_moving,2)
        current_obv = round(current_obv,2)
        current_qqqpastweek_percent_change = round(current_qqqpastweek_percent_change,2)
        
        #first_number = 0.0239*((final_sum)**2)-1.6689*(final_sum)+88.89
        #second_number = 0.0032*((final_expected_value)**2)-.1382*(final_expected_value)+62.704
        #final_number = (first_number+second_number)/2
        #final_number = round(final_number,2)
        final_number = 100*(0.0238*(final_sum)-.7306)
        final_number = round(final_number,2)

        print(beginning_date, final_number,"%",nextweekchange)
        
    else:
        pass