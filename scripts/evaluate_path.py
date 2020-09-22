import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import signal


PATH_TO_TEXT_FILES = '/home/tfpc/bag_records/text_files/'
PATH_TO_CSV_FILE = '/home/tfpc/bag_records/example.csv'
PATH_TO_PLOTS = PATH_TO_TEXT_FILES + 'plots'


def initialize_dataframes(gt_df, fp_df):
    start_time = fp_df['timestamp'][0]
    gt_df = gt_df[gt_df['timestamp'] > start_time]
    gt_df['timestamp'] -= start_time
    fp_df['timestamp'] -= start_time
    gt_len = len(gt_df)
    fp_len = len(fp_df)
    if gt_len > fp_len:
        gt_df.drop(gt_df.tail(gt_len - fp_len).index, inplace=True)
    elif gt_len < fp_len:
        fp_df.drop(fp_df.tail(fp_len - gt_len).index, inplace=True)
    return gt_df, fp_df


def plot_column_over_time(x_axis, y_axis, df_1=None, df_2=None, mean=None, std=None):
    x_unit = get_unit(x_axis)
    y_unit = get_unit(y_axis)
    if df_1 is None and df_2 is None:
        print("Error: not dataframes had been sent")
        return
    fig, ax1 = plt.subplots(figsize=(30,20))
    ax1.set_title('{} vs. {}'.format(y_axis, x_axis), fontsize=30)
    if df_1 is not None:
        plt.plot(df_1[x_axis], df_1[y_axis], color='b', label="Ground Truth")
    if df_2 is not None:
        plt.plot(df_2[x_axis], df_2[y_axis], color='r', label="Planned Path")
    ax1.set_xlabel('{}[{}]'.format(x_axis, x_unit), fontsize=25)
    ax1.set_ylabel('{}[{}]'.format(y_axis, y_unit), fontsize=25)
    ax1.legend(loc=1, prop={'size': 25})
    plt.savefig(PATH_TO_PLOTS + '/{} over {}'.format(y_axis, x_axis))


def plot_error_over_time(x_axis, y_axis, df_1):
    x_unit = get_unit(x_axis)
    y_unit = get_unit(y_axis)
    fig, ax1 = plt.subplots(figsize=(30, 20))
    ax1.set_title('{} error vs. {}'.format(y_axis, x_axis), fontsize=30)
    plt.plot(df_1[x_axis], df_1[y_axis], color='b')
    ax1.set_xlabel('{}[{}]'.format(x_axis, x_unit), fontsize=25)
    ax1.set_ylabel('{}[{}]'.format(y_axis, y_unit), fontsize=25)
#     ax1.legend(loc=1, prop={'size': 25})
    plt.savefig(PATH_TO_PLOTS + '/{} error'.format(y_axis))


def get_unit(axis):
    if axis == 'timestamp':
        return 'sec'
    if axis in ['x', 'y', 'z']:
        return 'm'
    elif axis in ['vx', 'vy', 'vz']:
        return 'm/sec'
    elif axis == 'yaw':
        return 'deg'
    elif axis == 'yaw_dot':
        return 'deg/sec'


def get_stats(df, column):
    return df[column].mean(), df[column].std(), df[column].max()


def set_euclidean_error(df, axis1, axis2, axis3=None):
    if axis3 is None:
        return np.sqrt(df[axis1] ** 2 + df[axis2] ** 2)
    else:
        return np.sqrt(df[axis1] ** 2 + df[axis2] ** 2 + df[axis3] ** 2)


def create_evaluate_df(gt_df, fp_df):
    evaluate_df = gt_df - fp_df
    evaluate_df['timestamp'] = gt_df['timestamp']

    evaluate_df['euclidean'] = set_euclidean_error(evaluate_df, 'x', 'y', 'z')
    evaluate_df['xy_euclidean'] = set_euclidean_error(evaluate_df, 'x', 'y')
    evaluate_df['vel_euclidean'] = set_euclidean_error(evaluate_df, 'vx', 'vy', 'vz')
    evaluate_df['xy_vel_euclidean'] = set_euclidean_error(evaluate_df, 'vx', 'vy')
    return evaluate_df


def plot_path_and_errors(gt_df, fp_df, evaluate_df, stop_index):
    for column in gt_df:
        if column == 'timestamp':
            continue
        plot_column_over_time('timestamp', column, gt_df, fp_df)

    for column in evaluate_df:
        if column == 'timestamp':
            continue
        plot_error_over_time('timestamp', column, evaluate_df)

    plot_column_over_time('timestamp', 'euclidean', evaluate_df[stop_index:])
    plot_column_over_time('timestamp', 'vel_euclidean', evaluate_df[stop_index:])


def find_delay(data_x, data_y, radius=200):
    data_x = data_x.to_numpy()
    data_y = data_y.to_numpy()
    result = np.zeros(radius)
    for n in range(radius):
        if n == 0:
            diff = data_x - data_y
        else:
            diff = data_x[:-n] - data_y[n:]
        result[n] = np.mean(np.abs(diff))
    return np.argmin(result)


gt_df = pd.read_csv(PATH_TO_TEXT_FILES + "stamped_groundtruth.txt", sep=" ")
fp_df = pd.read_csv(PATH_TO_TEXT_FILES + "stamped_traj_estimate.txt", sep=" ")
errors_df = pd.read_csv(PATH_TO_CSV_FILE)

if not os.path.exists(PATH_TO_PLOTS):
    os.makedirs(PATH_TO_PLOTS)

gt_df, fp_df = initialize_dataframes(gt_df, fp_df)
stop_index = fp_df[(fp_df[['vx', 'vy', 'vz']] == 0).all(1)].index[0]

evaluate_df = create_evaluate_df(gt_df, fp_df)

# plot_column_over_time('x', 'y', gt_df, fp_df)
#
# plot_path_and_errors(gt_df, fp_df, evaluate_df, stop_index)

stop_index = fp_df[(fp_df[['vx', 'vy', 'vz']] == 0).all(1)].index[0]
errors_stats = {}
for column in evaluate_df[['euclidean', 'xy_euclidean', 'vel_euclidean', 'xy_vel_euclidean', 'yaw', 'yaw_dot']]:
    errors_stats[column] = get_stats(evaluate_df, column)
errors_stats['stop_euclidean'] = get_stats(evaluate_df[stop_index:], 'euclidean')
errors_stats['stop_vel'] = get_stats(evaluate_df[stop_index:], 'vel_euclidean')

errors_ls = []
for k in errors_stats:
    errors_ls += errors_stats[k]


axes = ['z', 'yaw']
dt = fp_df['timestamp'][1] - fp_df['timestamp'][0]

for axis in axes:
    lag = find_delay(fp_df[axis], gt_df[axis])
    errors_ls.append(fp_df['timestamp'][lag])

a_series = pd.Series(errors_ls, index=errors_df.columns)
errors_df = errors_df.append(a_series, ignore_index=True)
errors_df.to_csv(PATH_TO_CSV_FILE, index=False)


def cross_corr(y1, y2):
  """Calculates the cross correlation and lags without normalization.

  The definition of the discrete cross-correlation is in:
  https://www.mathworks.com/help/matlab/ref/xcorr.html

  Args:
    y1, y2: Should have the same length.

  Returns:
    max_corr: Maximum correlation without normalization.
    lag: The lag in terms of the index.
  """
  if len(y1) != len(y2):
    raise ValueError('The lengths of the inputs should be the same.')

  y1_auto_corr = np.dot(y1, y1) / len(y1)
  y2_auto_corr = np.dot(y2, y2) / len(y1)
  corr = np.correlate(y1, y2, mode='same')
  # The unbiased sample size is N - lag.
  unbiased_sample_size = np.correlate(
      np.ones(len(y1)), np.ones(len(y1)), mode='same')
  corr = corr / unbiased_sample_size / np.sqrt(y1_auto_corr * y2_auto_corr)
  shift = len(y1) // 2

  max_corr = np.max(corr)
  argmax_corr = np.argmax(corr)
  return max_corr, argmax_corr - shift, corr[20]







