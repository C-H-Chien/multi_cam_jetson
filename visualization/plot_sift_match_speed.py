import os
import numpy as np
import matplotlib.pyplot as plt
from viz_utils import linear_regression

image_size = "640x400"
power_mode = "15w"
num_of_available_cameras = 4

current_working_directory = os.getcwd()
data_path = os.path.join(current_working_directory, "../data/sift_match_GPU_speed/" + image_size + "_" + power_mode + "/")

# Load data of the speed of matching SIFT features on GPU using four Arducam cameras 
breakdown_timings = [];
for i in range(num_of_available_cameras):
    data = np.loadtxt(os.path.join(data_path, "arducam_" + str(i+1) + "_pair.txt"), delimiter="\t")
    breakdown_timings.append(data)

# Name of each column in order: initialization time, enqueue time, processing time, cleanup time, 
# total GPU extraction time on frame 1, total GPU extraction time on frame 2, total GPU matching time, total batch time
breakdown_timings_dict_list = []
for i in range(num_of_available_cameras):  
    breakdown_timings_dict = {}
    sift_warmup_time = breakdown_timings[i][:, 4] - breakdown_timings[i][:, 5]
    breakdown_timings_dict["initialization"] = breakdown_timings[i][:, 0] + sift_warmup_time
    breakdown_timings_dict["extraction_(image 1)"] = breakdown_timings[i][:, 4] - np.mean(sift_warmup_time, axis=0)
    breakdown_timings_dict["extraction_(image 2)"] = breakdown_timings[i][:, 5]
    breakdown_timings_dict["matching"] = breakdown_timings[i][:, 6]
    breakdown_timings_dict["others"] = breakdown_timings[i][:, 1]
    breakdown_timings_dict["cleanup"] = breakdown_timings[i][:, 3]
    breakdown_timings_dict_list.append(breakdown_timings_dict)

# compute mean and std of each column of the breakdown_timings for all img_pair_index
means_list = []
stds_list = []
for img_pair_index in range(num_of_available_cameras):
    means_dict = {}
    stds_dict = {}
    timing_dict = breakdown_timings_dict_list[img_pair_index]
    
    for key in timing_dict.keys():
        means_dict[key] = np.mean(timing_dict[key])
        stds_dict[key] = np.std(timing_dict[key])
    
    means_list.append(means_dict)
    stds_list.append(stds_dict)

img_pair_index = 0
init_key = "initialization"
ext1_key = "extraction_(image 1)"
ext2_key = "extraction_(image 2)"
match_key = "matching"
others_key = "others"
cleanup_key = "cleanup"

# print(f"(Mean, Std) of initialization time: {means_list[img_pair_index][init_key]}, {stds_list[img_pair_index][init_key]}")
# print(f"(Mean, Std) of SIFT extraction (frame 1) time: {means_list[img_pair_index][ext1_key]}, {stds_list[img_pair_index][ext1_key]}")
# print(f"(Mean, Std) of SIFT extraction (frame 2) time: {means_list[img_pair_index][ext2_key]}, {stds_list[img_pair_index][ext2_key]}")
# print(f"(Mean, Std) of SIFT matching time: {means_list[img_pair_index][match_key]}, {stds_list[img_pair_index][match_key]}")
# print(f"(Mean, Std) of others time: {means_list[img_pair_index][others_key]}, {stds_list[img_pair_index][others_key]}")
# print(f"(Mean, Std) of cleanup time: {means_list[img_pair_index][cleanup_key]}, {stds_list[img_pair_index][cleanup_key]}")

# Plot the breakdown timings of each column
# make labels break line in the middle of the word
# timing_dict = breakdown_timings_dict_list[img_pair_index]
# data_array = np.array(list(timing_dict.values()))
# labels = [label.replace("_", "\n") for label in timing_dict.keys()]
# plt.boxplot(data_array.T, labels=labels)
# plt.title(f"Breakdown timings for matching a single image pair ({image_size}) on {power_mode} mode")
# plt.xlabel("stages")
# plt.ylabel("time (ms)")
# plt.show()

all_cameras = np.array([1, 2, 3, 4, 5, 6, 7, 8])
x_existing = all_cameras[:num_of_available_cameras]  # Number of image pairs
x_prediction = all_cameras[num_of_available_cameras:]  # Points to predict

# Prepare data for each segment
segments = {
    'initialization': [means_list[i][init_key] for i in range(num_of_available_cameras)],
    'extraction': [means_list[i][ext1_key] + means_list[i][ext2_key] for i in range(num_of_available_cameras)],
    'matching': [means_list[i][match_key] for i in range(num_of_available_cameras)],
    'others': [means_list[i][others_key] for i in range(num_of_available_cameras)],
    'cleanup': [means_list[i][cleanup_key] for i in range(num_of_available_cameras)]
}



extrapolated_values = {}
for segment_name, y_existing in segments.items():
    # Fit linear regression
    slope, intercept = linear_regression(x_existing, y_existing)
    
    # Predict for 5-8 image pairs
    y_predicted = slope * x_prediction + intercept
    extrapolated_values[segment_name] = y_predicted

# Combine existing and extrapolated data
category = ["1", "2", "3", "4", "5", "6", "7", "8"]
values_segment1 = segments['initialization'] + list(extrapolated_values['initialization'])
values_segment2 = segments['extraction'] + list(extrapolated_values['extraction'])
values_segment3 = segments['matching'] + list(extrapolated_values['matching'])
values_segment4 = segments['others'] + list(extrapolated_values['others'])
values_segment5 = segments['cleanup'] + list(extrapolated_values['cleanup'])

# Create stacked bar chart with visual distinction between actual and predicted data
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors_predicted = ['#87ceeb', '#ffb347', '#90ee90', '#ff6b6b', '#dda0dd']

# Plot actual data
# combine the following code to one bar chart
plt.bar(category, values_segment1, label='Initialization', color=colors[0])
plt.bar(category, values_segment2, bottom=values_segment1, label='SIFT extraction', color=colors[1])
plt.bar(category, values_segment3, bottom=[s1 + s2 for s1, s2 in zip(values_segment1, values_segment2)], label='SIFT matching', color=colors[2])
plt.bar(category, values_segment4, bottom=[s1 + s2 + s3 for s1, s2, s3 in zip(values_segment1, values_segment2, values_segment3)], label='Others', color=colors[3])
plt.bar(category, values_segment5, bottom=[s1 + s2 + s3 + s4 for s1, s2, s3, s4 in zip(values_segment1, values_segment2, values_segment3, values_segment4)], label='Cleanup', color=colors[4])

# add numbers on top of each segment
for i in range(8):
    # print(f"values_segment1[i]: {values_segment1[i]}, values_segment2[i]: {values_segment2[i]}, values_segment3[i]: {values_segment3[i]}, values_segment4[i]: {values_segment4[i]}, values_segment5[i]: {values_segment5[i]}")
    plt.text(i, values_segment1[i] * 0.5, f'{values_segment1[i]:.2f}', ha='center', va='center')
    plt.text(i, values_segment1[i] + values_segment2[i] * 0.5, f'{values_segment2[i]:.2f}', ha='center', va='center')
    plt.text(i, values_segment1[i] + values_segment2[i] + values_segment3[i] * 0.5, f'{values_segment3[i]:.2f}', ha='center', va='center')
    # plt.text(i, values_segment1[i] + values_segment2[i] + values_segment3[i] + values_segment4[i] * 0.5, f'{values_segment4[i]:.2f}', ha='center', va='center')
    plt.text(i, values_segment1[i] + values_segment2[i] + values_segment3[i] + values_segment5[i] * 0.5, f'{values_segment5[i]:.2f}', ha='center', va='center')
    # add a total number on top of each bar and make boldfaced
    plt.text(i, 10 + values_segment1[i] + values_segment2[i] + values_segment3[i] + values_segment5[i], f'{values_segment1[i] + values_segment2[i] + values_segment3[i] + values_segment5[i]:.2f}', ha='center', va='center', fontweight='bold')

# # Plot predicted data with different colors
# plt.bar(category[8:], values_segment1[8:], color=colors_predicted[0], alpha=0.7)
# plt.bar(category[8:], values_segment2[8:], bottom=values_segment1[4:], color=colors_predicted[1], alpha=0.7)
# plt.bar(category[8:], values_segment3[8:], bottom=[s1 + s2 for s1, s2 in zip(values_segment1[4:], values_segment2[4:])], color=colors_predicted[2], alpha=0.7)
# plt.bar(category[8:], values_segment4[8:], bottom=[s1 + s2 + s3 for s1, s2, s3 in zip(values_segment1[4:], values_segment2[4:], values_segment3[4:])], color=colors_predicted[3], alpha=0.7)
# plt.bar(category[8:], values_segment5[8:], bottom=[s1 + s2 + s3 + s4 for s1, s2, s3, s4 in zip(values_segment1[4:], values_segment2[4:], values_segment3[4:], values_segment4[4:])], color=colors_predicted[4], alpha=0.7)

# # Add vertical line to separate actual from predicted data
# plt.axvline(x=3.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
# plt.text(3.5, plt.ylim()[1] * 0.95, 'Predicted', rotation=90, va='top', ha='right', color='red', fontweight='bold')

# change the font size of the x and y labels and the title
plt.xlabel('Number of image pairs', fontsize=14)
plt.ylabel('Time (ms)', fontsize=14)
plt.title(f"Breakdown timings for matching image pairs ({image_size}) on {power_mode} mode", fontsize=14)
# plt.ylim(0, 3500)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

