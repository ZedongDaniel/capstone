import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

test_data = pd.read_csv('test_set.csv', parse_dates=['date'])
test_data.set_index('date', inplace=True)

def __get_models(anomaly_dir):
    models = list(anomaly_dir.keys())
    return models


def __get_sectors(df = test_data):
    sectors = list(df.columns)
    return sectors


def summarize_anomalies(anomaly_dir, print_info = False, plot = False):
    '''
    Print the statistics of anomalies detected per sector by each model

    Returns:
    - anomaly_counts: DataFrame
        counts the number of anomalies detected per sector and model
    - anomaly_percentage: DataFrame
        shows the percentage of anomalies over total number of observations per sector
    '''
    models = __get_models(anomaly_dir)
    sectors = __get_sectors()
    anomaly_counts = pd.DataFrame(index = sectors, columns = models)
    for model in models:
        anomaly_df = anomaly_dir[model]
        anomaly_counts[model] = anomaly_df.sum()

    num_obs = test_data.shape[0]
    anomaly_percentage = (anomaly_counts / num_obs) * 100

    if print_info:
        print('Anomalies Count per Sector and Model: ')
        print(anomaly_counts)
        print("\n")
        print('Anomalies Percentage per Sector Over Total Number of Observation: ')
        print(anomaly_percentage)

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(anomaly_percentage.astype(float), annot=True, fmt=".2f", cmap='Blues')
        plt.title('Percentage of Anomalies Detected per Sector by Each Model')
        plt.xlabel('Model')
        plt.ylabel('Sector')
        plt.show()

    return anomaly_counts, anomaly_percentage


# Plot the anomalies points on the Log-return per sector for one Model
def plot_anomalies(anomaly_dir, model_name, log_returns = test_data):
    anomaly_df = anomaly_dir[model_name]

    fig, axes = plt.subplots(4, 3, figsize=(16, 10))
    fig.suptitle(f'Anomalies by Sectors ({model_name})')
    fig.tight_layout(pad=3.0)

    for i, sector in enumerate(anomaly_df.columns):
        ax = axes.flatten()[i]

        sector_test = log_returns[sector]

        ax.plot(sector_test.index, sector_test.values, label=f"{sector} Test Data")

        anomalies = anomaly_df[sector]
        anomalies_dates = anomalies[anomalies == 1].index
        anomalies_values = log_returns.loc[anomalies_dates, sector]

        ax.scatter(anomalies_dates, anomalies_values, color = 'r', marker = 'o', label = 'Anomaly')

        ax.set_title(f'{sector}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Log Return')
        ax.legend()

    plt.tight_layout()
    plt.show()


# Plot the distribution of anomalies per models 
def plot_anomaly_distribution(anomaly_dir):
    models = __get_models(anomaly_dir)
    fig, axes = plt.subplots(len(models), 1, figsize = (10, 2.5 * len(models)))

    for i, model in enumerate(models):
        anomalies = anomaly_dir[model]
        anomalies_per_date = anomalies.sum(axis=1)

        ax = axes[i]
        anomalies_per_date.plot(ax=ax)
        ax.set_title(f'Number of Anomalies Detected Over Time - {model}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Anomalies')

    plt.tight_layout()
    plt.show()


# Shows the correlations between anomalies per models
def jaccard_similarity(anomalies1, anomalies2):
    anomalies1_flat = anomalies1.values.flatten()
    anomalies2_flat = anomalies2.values.flatten()
    intersection = np.logical_and(anomalies1_flat == 1, anomalies2_flat == 1).sum()
    union = np.logical_or(anomalies1_flat == 1, anomalies2_flat == 1).sum()
    if union == 0:
        return np.nan
    return intersection / union

def get_jaccard(anomaly_dir):
    models = __get_models(anomaly_dir)
    jaccard_df = pd.DataFrame(index = models, columns = models)

    for model1, model2 in combinations(models, 2):
        sim = jaccard_similarity(anomaly_dir[model1], anomaly_dir[model2])
        jaccard_df.loc[model1, model2] = sim
        jaccard_df.loc[model2, model1] = sim

    np.fill_diagonal(jaccard_df.values, 1.0)

    print("Jaccard Similarity Index Between Models:")

    plt.figure(figsize=(8, 6))
    sns.heatmap(jaccard_df.astype(float), annot=True, cmap='Blues', fmt=".2f")
    plt.title('Jaccard Similarity Index Between Models')
    plt.show()


# Get Anomalies Consensus
def anomalies_consensus(anomaly_dir, model_list, plot = False, log_return = test_data):
    sectors = __get_sectors(log_return)
    models = __get_models(anomaly_dir)

    sector_anomaly_consensus = {}

    print('Getting Consensus Anomalies for Models: ', model_list)

    for sector in sectors:
        anomaly_flat_sector = pd.DataFrame(index = log_return.index)

        for model in model_list:
            anomaly_flat_sector[model] = anomaly_dir[model][sector]

        anomaly_sum_sector = anomaly_flat_sector.sum(axis=1)
        sector_anomaly_consensus[sector] = (anomaly_sum_sector > (len(model_list) - 1)).astype(int)
        sector_anomaly_consensus = pd.DataFrame(sector_anomaly_consensus)
    
    model_name = '+'.join(model_list)

    if plot:
        anomaly_cons_dict = {}
        anomaly_cons_dict[model_name] = sector_anomaly_consensus
        plot_anomalies(anomaly_cons_dict, model_name)

    return sector_anomaly_consensus


# Majority Voting Anomalies Consensus
def majority_anomalies_consensus(anomaly_dir, consensus_threshold = 4, plot = False, log_return = test_data):
    sectors = __get_sectors(log_return)
    models = __get_models(anomaly_dir)

    majority_anomaly_consensus = {}

    for sector in sectors:
        anomaly_flat_sector = pd.DataFrame(index = log_return.index)

        for model in models:
            anomaly_flat_sector[model] = anomaly_dir[model][sector]

        anomaly_sum_sector = anomaly_flat_sector.sum(axis=1)
        majority_anomaly_consensus[sector] = (anomaly_sum_sector >= consensus_threshold).astype(int)
        majority_anomaly_consensus = pd.DataFrame(majority_anomaly_consensus)

    if plot:
        anomaly_cons_dict = {}
        anomaly_cons_dict['Majority Consensus'] = majority_anomaly_consensus
        plot_anomalies(anomaly_cons_dict, 'Majority Consensus')

    return majority_anomaly_consensus


# Return a dict of anomaly dates
def anomaly_dates(anomaly_df):
    anomaly_dates_dict = {}
    sectors = anomaly_df.columns

    for sector in sectors:
        sector_anomaly = anomaly_df[sector]
        sector_anomaly_dates = sector_anomaly[sector_anomaly == 1].index.tolist()
        anomaly_dates_dict[sector] = sector_anomaly_dates

    return anomaly_dates_dict

    
# Return anomaly period
def anomaly_period(anomaly_df):
    '''
    Input:
    - anomaly_df: DataFrame
        
    Returns: 
    - anomaly_period: List
        with element being TimeStamp1:TimeStamp2 indicating one Anomaly period
        
    Anomaly Period is defined as the time span starting from the first detected 
    anomaly and ending when no anomalies occur for three consecutive days.
    '''
    anomaly_period_dict = {}
    dates = anomaly_df.index

    for sector in anomaly_df.columns:
        sector_anomaly = anomaly_df[sector]
        periods = []
        in_anomaly_period = False
        start_date = None
        no_anomaly_count = 0

        for i in range(len(sector_anomaly)):
            date = dates[i]
            anomaly = sector_anomaly.iloc[i]

            if not in_anomaly_period:
                if anomaly == 1:
                    in_anomaly_period = True
                    start_date = date
                    no_anomaly_count = 0

            else:
                if anomaly == 1:
                    no_anomaly_count = 0
                else:
                    no_anomaly_count += 1
                    if no_anomaly_count >= 5:
                        end_index = i - no_anomaly_count
                        end_date = dates[end_index]
                        periods.append((start_date, end_date))
                        in_anomaly_period = False
                        no_anomaly_count = 0
                        start_date = None

        if in_anomaly_period == True:
            end_date = dates[-1]
            periods.append((start_date, end_date))

        anomaly_period_dict[sector] = periods
    
    return anomaly_period_dict
    
                
# Identify Macro Economic period  
def macro_anomalies_period(anomaly_consensus_df, threshold = 7):
    total_anomalies = anomaly_consensus_df.sum(axis = 1)

    macro_anomalies = pd.DataFrame((total_anomalies >= threshold).astype(int), columns=['Anomaly'])

    macro_economic_period = anomaly_period(macro_anomalies)['Anomaly']

    return macro_economic_period


# Identify Sector Specific Anomaly Period
def sector_specific_period(anomaly_period_dict, macro_anomaly_periods):
    sectors = list(anomaly_period_dict.keys())

    sector_specific_periods = {}

    for sector in sectors:
        sector_periods = anomaly_period_dict[sector]
        specific_periods = []
        for sector_interval in sector_periods:
            remaining_intervals = subtract_interval_helper(sector_interval, macro_anomaly_periods)
            specific_periods.extend(remaining_intervals)
        sector_specific_periods[sector] = specific_periods

    return sector_specific_periods
    
def subtract_interval_helper(interval, intervals_to_sub):
    result_intervals = [interval]

    for sub_interval in intervals_to_sub:
        new_result_intervals = []

        for current_interval in result_intervals:
            a_start, a_end = current_interval
            b_start, b_end = sub_interval

            if a_end < b_start or b_end < a_start:
                new_result_intervals.append(current_interval)

            else:
                if a_start < b_start:
                    new_result_intervals.append((a_start, b_start - pd.Timedelta(days = 1)))
                    if a_end > b_end:
                        new_result_intervals.append((b_end + pd.Timedelta(days = 1), a_end))
                else:
                    if a_end > b_end:
                        new_result_intervals.append((b_end + pd. Timedelta(days = 1), a_end))
                
        result_intervals = new_result_intervals

    return result_intervals


# # Macro Anomaly Period
# def macro_anomaly_period(anomaly_period_dict):
#     if not anomaly_period_dict:
#         return []

#     sectors = list(anomaly_period_dict.keys())

#     common_period = anomaly_period_dict[sectors[0]]

#     common_interval = common_period.copy()

#     for sector in sectors[1:]:
#         sector_intervals = anomaly_period_dict[sector]
#         new_common_interval = []

#         for interval1 in common_interval:
#             for interval2 in sector_intervals:
#                 start = max(interval1[0], interval2[0])
#                 end = min(interval1[1], interval2[1])

#                 if start <= end:
#                     new_common_interval.append((start, end))

#         common_interval = new_common_interval.copy()

#         if not common_interval:
#             return []

#     macro_economic_period = merge_overlapping_interval(common_interval) 

#     return macro_economic_period

# def merge_overlapping_interval(intervals):
#     if not intervals:
#         return []

#     intervals.sort(key= lambda x: x[0])

#     merged_intervals = [intervals[0]]

#     for current in intervals[1:]:
#         previous = merged_intervals[-1]

#         if current[0] <= previous[1]:
#             merged = (previous[0], max(previous[1], current[1]))
#             merged_intervals[-1] = merged
#         else:
#             merged_intervals.append(current)
    
#     return merged_intervals





    



