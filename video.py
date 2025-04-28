import cv2
import torch
from config import get_config
from argparse import Namespace
import pandas as pd
from resnet import get_resnet


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
        print(t)
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





class Video():

    def __init__(self, args: Namespace)-> None:
        self.config = get_config(args.video)
        self.resnet = get_resnet(args)
        self.resnet.resnet.to(device)

        if self.config["live"] == False:
            self.make_dataset()
        else:
            Exception("Live video not supported yet")


    def make_dataset(self)-> None:
        self.videos = get_config(self.config["videos"])["entries"]

        for file in self.videos:
            cap: cv2.VideoCapture = cv2.VideoCapture(file["file"])
            fps = cap.get(cv2.CAP_PROP_FPS)
            dataframe = read_video(cap, fps, file["score"], self.resnet)
            print(dataframe)






def get_video(args: Namespace) -> Video:
    return Video(args)



if __name__ == "__main__":
    args = Namespace()
    args.video = "video.json"
    args.resnet = "resnet.json"
    video = get_video(args)
    print(video)
