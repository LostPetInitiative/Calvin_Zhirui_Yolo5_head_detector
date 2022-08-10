import cv2

def detect_head_and_body(model, img):
    # print("img")
    # print(img.shape)
    # print(img.dtype)
    if img is None:
        return None, None, None
    annotated_img = img.copy()
    pred = model(img)
    # print("prediction")
    # print(pred)
    if len(pred.xyxy) > 0:

        xyxy = pred.xyxy[0].cpu().numpy()

        body = xyxy[xyxy[:, 5] == 1, :4].astype(int)

        # bodies_count = body.shape[0]
        # print(f"bodies detected: {bodies_count}")    
        
        # if len(body) > 0:
        #     body = [body[:, 0].min(), body[:, 1].min(), body[:, 2].max(), body[:, 3].max()]
        #     body_crop = img[body[1]: body[3], body[0]:body[2], :]
        #     annotated_img = cv2.rectangle(annotated_img, (body[0], body[1]), (body[2], body[3]), (255, 0, 0), 1) # color is in BGR
        # else:
        #     body_crop = img

        head = xyxy[xyxy[:, 5] == 0, :4].astype(int)

        heads_count = head.shape[0]
        # print(f"heads detected: {heads_count}")

        if len(head) > 0:
            head = [head[:, 0].min(), head[:, 1].min(), head[:, 2].max(), head[:, 3].max()]
            head_crop = img[head[1]: head[3], head[0]:head[2], :]
            annotated_img = cv2.rectangle(annotated_img, (head[0], head[1]), (head[2], head[3]), (255, 255, 0), 1) # color is in BGR
        else:
            head_crop = img

        return head_crop, annotated_img, heads_count
    else:        
        return img, annotated_img, 0
