
import os


from glob import glob

import numpy as np

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        print("setup")

    def predict(self, 
        video: Path = Input(description="input video", default=None),
        target_prompts: str = Input(
            description='prompts to change the video to',
            default='a panda surfing\na cartoon sloth surfing',
        ),  
        video_prompt: str = Input(
            description='prompts describing the original video',
            default='a man surfing'),
        ) -> Path:
        print("predict")
        os.system("rm -rf /outputs")
        os.system(f'accelerate launch train_tuneavideo.py --config="configs/replicate.yaml" --video-path {str(video)} --target-prompts "{target_prompts}" --video-prompt "{video_prompt}" --output-dir "/outputs"')
        os.system("ls -l /outputs")

        # find last file in path with .gif extension
        gif_path = max(glob('/outputs/*.gif'), key=os.path.getctime)
        return Path(gif_path)

