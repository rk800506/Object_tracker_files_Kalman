# the directory are relative to the parent dataset dir
# not_raw in UAV123 is not same as not_raw in visdrone or DTB70
# root_dir have been given

### UAV123 ###
root_dir = "/media/dtu-project2/2GB_HDD/object_tracker/UAV123/CFtracker_output_opencv2"
overall_SR              not_raw
overall_P		        not_raw
full_occlusion_SR       not_raw/specificallly_for_precision
full_occlusion_P	    full_occlusion_precision_plot_data
partial_occlusion_SR	not_raw
partial_occlusion_P	    partial_occlusion_precision_plot_data
low_resolution_SR	    low_res_success_plot_data
low_resolution_P	    low_res_precision_plot_data

### visdrone ####
root_dir = "/media/dtu-project2/2GB_HDD/object_tracker/visdrone/CFtracker_output_opencv"
overall_SR		        not_raw
overall_P		        not_raw
full_occlusion_SR	    not_raw
full_occlusion_P	    not_raw
partial_occlusion_SR    success_rate_partial_occlusion_data
partial_occlusion_P	    precision_partial_occlusion_data
low_resolution_SR	    success_rate_plot_low_res_data
low_resolution_P	    success_rate_plot_low_res_data

### DTB70 ### 
root_dir = "/media/dtu-project2/2GB_HDD/object_tracker/DTB70/CFtracker_output_opencv"
overall_SR		        not_raw
overall_P		        not_raw
low_resolution_SR	    low_res_success_rate
low_resolution_P	    low_res_precision
occlusion_SR		    occlusion_success_rate
occlusion_P		        occlusion_precision
