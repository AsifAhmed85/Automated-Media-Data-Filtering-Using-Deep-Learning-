import os

video_class = 2
# Loop through the filesystem
for root, dirs, files in os.walk("/media/asif/Programming/Thesis/Video_Classifier/Dataset/Class_separated_video/Gaming"):
    # Loop through files
    files.sort()
    video_no = 1
    for name in files:
        # Consider only mp4
        if name.endswith('.mp4'):
            old_path = root + "/" + name
            new_name = f"{1:02}" + "_" + f"{video_class:02}" + "_" + f"{video_no:02}" + "_" + f"{0:02}" + '.mp4'
            new_path = root + "/" + new_name
            os.rename(old_path, new_path)
            video_no += 1
