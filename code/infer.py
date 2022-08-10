import kafkajobs

from model import detect_head_and_body

def infer_in_json_field(model, images_json_field):
    '''Takes serialized images and applied detection model for them'''

    imagesNp = kafkajobs.serialization.imagesFieldToNp(images_json_field)
    bodies = list()
    heads = list()
    annotations = list()
    body_counts = list()
    heads_counts = list()
    for imNumpy in imagesNp:                
        body, head, annotated, bodies_count, heads_count = detect_head_and_body(model, imNumpy)
        bodies.append(body)
        heads.append(head)
        annotations.append(annotated)
        body_counts.append(bodies_count)
        heads_counts.append(heads_count)
    bodies_enc = kafkajobs.serialization.imagesNpToStrList(bodies)
    heads_enc = kafkajobs.serialization.imagesNpToStrList(heads)
    annotations_enc = kafkajobs.serialization.imagesNpToStrList(annotations)
    
    yolo5_outputs = list()
    for i in range(0,len(bodies)):
        yolo5_output = dict()
        yolo5_output['body'] = bodies_enc[i]
        yolo5_output['head'] = heads_enc[i]
        yolo5_output['annotated'] = annotations_enc[i]
        yolo5_output['body_count'] = body_counts[i]
        yolo5_output['head_count'] = heads_counts[i]
        yolo5_outputs.append(yolo5_output)

    return yolo5_outputs
