import os

import cv2
import numpy as np


def read_video(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            # Display the frame in a window

            # cv2.imshow("Video Player", frame)
            frames.append(frame)

            # Press 'q' on the keyboard to exit the loop
            # cv.waitKey(25) determines the speed of playback (25ms delay per frame)

            # if cv2.waitKey(25) & 0xFF == ord("q"):
            # break
        else:
            # Break the loop if the video has ended or an error occurred
            break
    cap.release()
    cv2.destroyAllWindows()
    frames = np.array(frames)
    return frames


path = "./data/s1_processed/"

listdir = os.listdir(path)

file_path = ""
align_path = ""

one_person = []
print(len(listdir))
path_list = []
for i in range(len(listdir)):
    if i == 5:
        break
    file_path = os.path.join(path, listdir[i])
    if file_path == "./data/s1_processed/align":
        align_path = file_path
    else:
        frames = read_video(file_path)
        one_person.append(frames)
        path_list.append(file_path)

print(path_list)
one_person = np.array(one_person)
print(one_person.shape)

# align_path_ls = os.listdir(align_path)
# for i in range(len(align_path_ls)):
#     if i == 4:
#         break
#     align_path = align_path_ls[i]
#     print(align_path)


print(file_path)
frames = read_video(file_path)
print(frames.shape)
