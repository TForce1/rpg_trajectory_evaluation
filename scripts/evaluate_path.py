import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# df = pd.DataFrame(["/home/tfpc/bag_records/text_files/stamped_groundtruth.txt"])
gt_df = pd.read_csv("/home/tfpc/bag_records/text_files/stamped_groundtruth.txt", sep=" ")
fp_df = pd.read_csv("/home/tfpc/bag_records/text_files/stamped_traj_estimate.txt", sep=" ")
errors_df = pd.read_csv('/home/tfpc/bag_records/example.csv')

temp = gt_df['timestamp'][0]
gt_df['timestamp'] -= temp
fp_df['timestamp'] -= temp
gt_len = len(gt_df)
fp_len = len(fp_df)
if gt_len > fp_len:
    gt_df.drop(gt_df.tail(gt_len - fp_len).index,inplace=True)
elif gt_len < fp_len:
    fp_df.drop(fp_df.tail(fp_len - gt_len).index,inplace=True)


def plot_column_over_time(axis, unit, df_1=None, df_2=None, mean=None, std=None):
    if df_1 is None:
        print("Error: not dataframes had been sent")
        return
    fig, ax1 = plt.subplots(figsize=(30,20))
    #bar plot creation
    ax1.set_title('{} vs. Time'.format(axis), fontsize=30)
    sns.lineplot(x='timestamp', y=axis, data = df_1, color='b', label="Ground Truth")
    if df_2 is not None:
        sns.lineplot(ax=ax1, x='timestamp', y=axis, data = df_2, color='r', label="Planned Path")
#     if std is not None:
#         sns.lineplot(ax=ax1, x='timestamp', y=mean, data = df_2, color='magenta', label="Mean")
#     if std is not None:
#         sns.lineplot(ax=ax1, x='timestamp', y=std, data = df_2, color='g', label="STD")
    ax1.set_xlabel('Time[sec]', fontsize=25)
    ax1.set_ylabel('{}[{}]'.format(axis, unit), fontsize=25)
    ax1.legend(loc=1, prop={'size': 25})
    plt.show()


def get_stats(df, column):
    return df[column].mean(),df[column].std(),df[column].max()


fig = plt.figure(figsize=(30, 20))
plt.plot(gt_df['x'], gt_df['y'], label = 'x')
plt.plot(fp_df['x'], fp_df['y'], label = 'y')
plt.show()

evaluate_df = pd.DataFrame()
evaluate_df = gt_df - fp_df
evaluate_df['timestamp'] = gt_df['timestamp']
# evaluate_df['dyaw'] = evaluate_df['dyaw'].apply(lambda x: x if abs(x) <= 180 else (x + 360  if x < 180 else x - 360))

evaluate_df['odom_error'] = np.sqrt(evaluate_df['x'] ** 2 + evaluate_df['y'] ** 2 + evaluate_df['z'] ** 2)
evaluate_df['xy_odom_error'] = np.sqrt(evaluate_df['x'] ** 2 + evaluate_df['y'] ** 2)
evaluate_df['vel_error'] = np.sqrt(evaluate_df['vx'] ** 2 + evaluate_df['vy'] ** 2 + evaluate_df['vz'] ** 2)
evaluate_df['xy_vel_error'] = np.sqrt(evaluate_df['vx'] ** 2 + evaluate_df['vy'] ** 2)

plot_column_over_time('x', 'm', gt_df, fp_df)
plot_column_over_time('y', 'm', gt_df, fp_df)
plot_column_over_time('z', 'm', gt_df, fp_df)
plot_column_over_time('yaw', 'deg', gt_df, fp_df)
plot_column_over_time('vx', 'm/s', gt_df, fp_df)
plot_column_over_time('vy', 'm/s', gt_df, fp_df)
plot_column_over_time('vz', 'm/s', gt_df, fp_df)
plot_column_over_time('yaw_dot', 'deg/s', gt_df, fp_df)


errors_stats = {}
for column in evaluate_df[['odom_error', 'xy_odom_error', 'vel_error', 'xy_vel_error', 'yaw', 'yaw_dot']]:
    errors_stats[column] = get_stats(evaluate_df, column)

errors_ls = []
for k in errors_stats:
    errors_ls += errors_stats[k]


a_series = pd.Series(errors_ls, index = errors_df.columns)
errors_df = errors_df.append(a_series, ignore_index=True)

errors_df.to_csv('/home/tfpc/bag_records/example.csv', index=False)