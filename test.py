from moviepy.editor import VideoFileClip

file_dir = '/media/asif/Programming/Thesis/Video_Classifier/Code/mp4_clips/'
file_name = '1.mp4'
clip = VideoFileClip(file_dir+file_name)
print(clip.duration)
