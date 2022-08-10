import sys
import os
import unittest

from model import detect_head_and_body

import cv2
import torch


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

if __name__ == "__main__":
    unittest.main()
    
    print("Self test passed")