import sys
import os
import unittest

from model import detect_head_and_body
from infer import infer_in_json_field

import cv2
import torch

import json

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

sys.path.append("./yolov5/")
model = torch.hub.load('./yolov5/', 'custom', path='yolov5s.pt', source='local')
print("model loaded")

class TestDetector(unittest.TestCase):
    def test_detect_head_and_body(self):
        img = load_image("./example/thumb_Pet_1548608848.426671.jpg")

        body, head, annotated_img, bodies_count, heads_count = detect_head_and_body(model, img)

        print(f"head shape: {head.shape}")
        print(f"body shape: {body.shape}")
        assert bodies_count == 0
        assert heads_count == 1

        assert head.shape[0] == 131
        assert head.shape[1] == 112
        
        assert body.shape[0] == 400
        assert body.shape[1] == 400

    def test_detect_in_json_field(self):
        json_data_path = "./example/pet911ru_rl546808_distinct_photos_snapshot.json"
        expected_output_path = "./example/expected_output_pet911ru_rl546808.json"
        
        json_dict = json.load(open(json_data_path))
        images = json_dict["images"]
        yolo_output = infer_in_json_field(model, images)

        expected_output = json.load(open(expected_output_path))

        # assert json are equivalent
        assert json.dumps(yolo_output) == json.dumps(expected_output)


if __name__ == "__main__":
    unittest.main()
    
    print("Self test passed")