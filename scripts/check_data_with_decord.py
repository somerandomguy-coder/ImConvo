import os

import decord
import numpy as np
import tensorflow as tf
from decord import VideoReader, cpu, gpu

decord.bridge.set_bridge('tensorflow')

path = "./data/s1_processed/"

listdir = os.listdir(path)

file_path = ""
align_path = ""



class VideoGenerator:
    def __init__(self, video_dir, batch_size, frame_shape, n_frame, shuffle=False):
        self.video_dir = video_dir
        self.batch_size = batch_size
        self.frame_shape = frame_shape
        self.n_frame = n_frame
        self.shuffle = shuffle
        self.indices = np.arange(len(video_dir))
        self._on_epoch_end()
        self.video_paths = []


        # different stuff
        self.align_path = "" 
        listdir = os.listdir(video_dir)
        for i in range(len(listdir)):
            file_name = listdir[i]
            file_path = os.path.join(self.video_dir, file_name)
            
            if file_name == "align":
                self.align_path = file_path
            else: 
                self.video_paths.append(file_path)
        self.align_paths = []
        align_path_ls = os.listdir(self.align_path)
        for i in range(len(align_path_ls)):
            align_name = align_path_ls[i]
            align_path = os.path.join(self.align_path, align_name)
            self.align_paths.append(align_path)

            self._on_epoch_end()
   
    def _on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(
            len(self.video_paths) / self.batch_size
        )  # get the floor if n_video = 102, batch = 10, then the last 2 video will be skip, len() will be 10

    def _get_file_id_by_batch(self, id):
        file_paths = []
        align_paths = []
        for i in range(self.batch_size):
            file_index = self.batch_size * id + i
            file_paths.append(self.video_paths[self.indices[file_index]])
            align_paths.append(self.align_paths[self.indices[file_index]])
        return file_paths, align_paths

    def _get_label_from_align(self, filename) -> tf.Tensor:    
        with open(filename, "r") as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            linesplit = line.split()
            start = int(int(linesplit[0])/1000)
            end = int(int(linesplit[1])/1000)
            label = linesplit[2]
            # print(f"there's {end-start} {label}")
            for _ in range(end - start):
                labels.append(label)
            # print(label)
        labels = tf.convert_to_tensor(labels)
        return labels

    def __getitem__(self, batch_id):
        file_paths, align_paths = self._get_file_id_by_batch(batch_id)
        # VideoLoader might be a better choice for async sampling tho...          
        X = []
        y = []
        for file_path in file_paths:
            vr = VideoReader(file_path, ctx=cpu(0))

            X.append(vr)
        X = tf.convert_to_tensor(X)

        for align_path in align_paths:
            label = self._get_label_from_align(align_path)
            y.append(label)
        y = tf.convert_to_tensor(y)
        return X, y
            
