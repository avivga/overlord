import sys
import os
sys.path.insert(0, "stylegan2-pytorch")
sys.path.insert(0, "stylegan-encoder")
import tempfile
from pathlib import Path
import argparse
import imageio
import dlib
import PIL.Image
import numpy as np
import cog
from network.training import Model
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector


TARGET_AGE = {
    "All": 0,
    "0-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
    "70-79": 8,
}

PREDICTOR = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
LANDMARKS_DETECTOR = LandmarksDetector("shape_predictor_68_face_landmarks.dat")


class Predictor(cog.Predictor):
    def setup(self):

        manipulate_parser = argparse.ArgumentParser()
        manipulate_parser.add_argument("-bd", "--base-dir", type=str, default=".")
        manipulate_parser.add_argument(
            "-mn", "--model-name", type=str, default="overlord-ffhq-x256-age"
        )
        manipulate_parser.add_argument("-i", "--img-path", type=str)
        manipulate_parser.add_argument(
            "-r", "--reference-img-path", type=str, required=False
        )
        manipulate_parser.add_argument("-o", "--output-img-path", type=str, default="")
        self.args, self.extras = manipulate_parser.parse_known_args()

    @cog.input(
        "image",
        type=Path,
        help="input facial image. NOTE: image will be aligned and resized to 256*256",
    )
    @cog.input(
        "target_age",
        type=str,
        options=list(TARGET_AGE.keys()),
        default="All",
        help="output age",
    )
    def predict(self, image, target_age="All"):
        if os.path.isfile('aligned.png'):
            os.remove('aligned.png')
        if os.path.isfile('rgb_input.png'):
            os.remove('rgb_input.png')

        input_path = str(image)
        # webcam input might be rgba, convert to rgb first
        input = imageio.imread(input_path)
        if input.shape[-1] == 4:
            rgba_image = PIL.Image.open(input_path)
            rgb_image = rgba_image.convert('RGB')
            input_path = 'rgb_input.png'
            imageio.imwrite(input_path, rgb_image)

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        self.args.output_img_path = str(out_path)
        model_dir = "overlord-ffhq-x256-age"
        model = Model.load(model_dir)
        align_image(input_path, 'aligned.png')
        img = PIL.Image.open('aligned.png')
        img = np.array(img.resize((256, 256)))
        manipulated_imgs = model.manipulate_by_labels(img)
        manipulated_imgs_res = np.split(manipulated_imgs, 9, axis=1)
        if target_age == "All":
            res = np.concatenate(manipulated_imgs_res[1:], axis=0)

        else:
            res = manipulated_imgs_res[TARGET_AGE[target_age]]

        imageio.imwrite(self.args.output_img_path, res)
        return out_path


def align_image(raw_img_path, aligned_face_path):
    for i, face_landmarks in enumerate(LANDMARKS_DETECTOR.get_landmarks(raw_img_path), start=1):
        image_align(raw_img_path, aligned_face_path, face_landmarks)
