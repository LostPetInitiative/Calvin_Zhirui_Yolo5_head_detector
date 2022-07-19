import sys
import os

import cv2
import torch


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)



def crop_image(model, img):
    print("img")
    print(img.shape)
    print(img.dtype)
    if img is None:
        return None, None, None
    annotated_img = img.copy()
    pred = model(img)
    print("prediction")
    print(pred)
    if len(pred.xyxy) > 0:

        xyxy = pred.xyxy[0].cpu().numpy()

        body = xyxy[xyxy[:, 5] == 1, :4].astype(int)

        print(f"bodies detected: {body.shape[0]}")    

        if len(body) > 0:
            body = [body[:, 0].min(), body[:, 1].min(), body[:, 2].max(), body[:, 3].max()]
            body_crop = img[body[1]: body[3], body[0]:body[2], :]
            annotated_img = cv2.rectangle(annotated_img, (body[0], body[1]), (body[2], body[3]), (255, 0, 0), 1) # color is in BGR
        else:
            body_crop = img

        head = xyxy[xyxy[:, 5] == 0, :4].astype(int)

        print(f"heads detected: {head.shape[0]}")

        if len(head) > 0:
            head = [head[:, 0].min(), head[:, 1].min(), head[:, 2].max(), head[:, 3].max()]
            head_crop = img[head[1]: head[3], head[0]:head[2], :]
            annotated_img = cv2.rectangle(annotated_img, (head[0], head[1]), (head[2], head[3]), (255, 255, 0), 1) # color is in BGR
        else:
            head_crop = img

        return body_crop, head_crop, annotated_img
    else:        
        return img, img, annotated_img

if __name__ == "__main__":
    sys.path.append("./detection_model/yolov5/")
    model = torch.hub.load('./detection_model/yolov5/', 'custom', path='detection_model/data/yolov5s.pt', source='local')
    print("model loaded")

    img = load_image("./example/thumb_Pet_1548608848.426671.jpg")

    body, head, annotated_img = crop_image(model, img)

    os.makedirs("./example/output", exist_ok=True)
    save_image("./example/output/body.jpg", body)
    save_image("./example/output/head.jpg", head)
    save_image("./example/output/annotated.jpg", annotated_img)

    print("Done")