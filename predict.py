
import os


from glob import glob

import numpy as np

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        print("setup")

    def predict(self, 
        video: Path = Input(description="input video", default=None),
        source_prompt: str = Input(
            description='prompts describing the original video',
            default='a man surfing'),
        target_prompts: str = Input(
            description='prompts to change the video to',
            default='a panda surfing\na cartoon sloth surfing',
        ),  
        steps: int = Input(
            description='number of steps to train for',
            default=300,
        ),
        width: int = Input(
            description='width of the output video (multiples of 32)',
            default=512,
        ),
        height: int = Input(
            description='height of the output video (multiples of 32)',
            default=512,
        ),
        length: int = Input(
            description='length of the output video (in seconds)',
            default=5,
        ),
        ) -> Path:
        print("predict")
        os.system("rm -rf /outputs")
        os.system(f'accelerate launch train_tuneavideo.py --config="configs/replicate.yaml" --video-path {str(video)} --target-prompts "{target_prompts}" --video-prompt "{source_prompt}" --max-train-steps {steps} --width {width} --height {height} --video-length {length} --output-dir "/outputs"')
        os.system("ls -l /outputs")

        # find last file in path with .gif extension
        gif_path = max(glob('/outputs/samples/*.gif'), key=os.path.getctime)
        return Path(gif_path)

