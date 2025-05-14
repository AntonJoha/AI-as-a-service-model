import cv2
from typing import Any, List
from pandas.io.pytables import DataCol
import torch
import random
from config import get_config, get_pickle, make_pickle
from argparse import Namespace
import pandas as pd
from resnet import get_resnet, Resnet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shape_frame(frame, patch_size=224):
    height, width, _ = frame.shape
    patches = []

    # Loop over the image in steps of patch_size
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            # Calculate the region of interest (ROI)
            patch = frame[y:y+patch_size, x:x+patch_size]

            # Check if the patch is smaller than the patch_size (at edges)
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(torch.from_numpy(patch))
    return patches
    

def read_video(video, fps, score, resnet):
    frame_interval = fps
    frame_count = 0
    t = 0
    video_info = []

    while video.isOpened():
        torch.cuda.empty_cache()
        ret, frame = video.read()

        if not ret:
            break

        patches = shape_frame(frame)

        x = [resnet.forward(torch.permute(p, (2,1,0)).unsqueeze(0).to(device)).cpu() for p in patches]
        
        x = [torch.Tensor.numpy(i) for i in x]
        video_info.append((t, x, score))
        frame_count += 1
        t += 1 / fps
    video.release()

    return pd.DataFrame(
        video_info, columns=["time_info", "frame", "score"]
    )






class Video(torch.utils.data.Dataset):

    def __init__(self, args: Namespace, res:Resnet | None=None)-> None:
        self.config = get_config(args.video)
        if res is None:
            self.resnet: Resnet = get_resnet(args)
        else:
            self.resnet: Resnet = res
        self.resnet.to(device)

        if self.config["live"] == False:
            self.make_dataset()
        else:
            Exception("Live video not supported yet")


    def get_video(self, file: Any) -> pd.DataFrame:
        to_return = get_pickle(file["file"])
        if to_return is None:
            cap: cv2.VideoCapture = cv2.VideoCapture(file["file"])
            fps = cap.get(cv2.CAP_PROP_FPS)
            to_return= read_video(cap, fps, file["score"], self.resnet)
            make_pickle(file["file"], to_return)
        return to_return

    def make_dataset(self)-> None:
        self.videos = get_config(self.config["videos"])["entries"]

        self.data: List[pd.DataFrame] = []
        for file in self.videos:
            dataframe = self.get_video(file)

            self.data.append(dataframe)


    def __len__(self)-> int:
        count = 0
        for i in self.data:
                count += len(i.index)
        return count

    def __getitem__(self, index: int)-> Any:
        for i in self.data:
            if index < len(i.index):
                return i.iloc[index]["frame"][random.randint(0, len(i.iloc[index]["frame"])-1)], torch.scalar_tensor(float(i.iloc[index]["score"]), dtype=torch.float32)
            else:
                index -= len(i.index)
        raise IndexError("Index out of range")


def get_video(args: Namespace, res:Resnet| None =None) -> Video:
    return Video(args, res)



if __name__ == "__main__":
    args = Namespace()
    args.video = "video.json"
    args.resnet = "resnet.json"
    video = get_video(args)


    loader = torch.utils.data.DataLoader(video, batch_size=2, shuffle=False)

    for i, data in enumerate(loader):
        print("Hello: ", i, "Data", data[0], "Label", data[1])


    print(video)
