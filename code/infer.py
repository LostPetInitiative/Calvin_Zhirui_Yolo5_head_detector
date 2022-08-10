import os
import shutil
import copy

import kafkajobs

from model import detect_head_and_body

def infer_in_json_field(model, images_json_field):
    '''Takes serialized images and applied detection model for them'''

    imagesNp = kafkajobs.serialization.imagesFieldToNp(images_json_field)
    heads = list()
    annotations = list()
    heads_counts = list()
    for imNumpy in imagesNp:                
        head, annotated, heads_count = detect_head_and_body(model, imNumpy)
        heads.append(head)
        annotations.append(annotated)
        heads_counts.append(heads_count)    
    heads_enc = kafkajobs.serialization.imagesNpToStrList(heads)
    annotations_enc = kafkajobs.serialization.imagesNpToStrList(annotations)
    
    yolo5_outputs = list()
    for i in range(0,len(heads)):
        yolo5_output = dict()
        yolo5_output['head'] = heads_enc[i]
        yolo5_output['annotated'] = annotations_enc[i]
        yolo5_output['head_count'] = heads_counts[i]
        yolo5_outputs.append(yolo5_output)

    return yolo5_outputs

def process_job(model, job):
    workdir = '/tmp'

    uid = job["uid"]
    print("{0}: Starting to process the job".format(uid))
    images = job['images']
    print("{0}: Extracting {1} images".format(uid, len(images)))

    jobPath = os.path.join(workdir,uid)
    os.mkdir(jobPath)

    try:
        # deep copy job
        job_out = copy.deepcopy(job)        

        yolo5_outputs = infer_in_json_field(model, images)

        del job_out['images'] # saving some kafka space
        job_out['yolo5_output'] = yolo5_outputs

        return job_out, uid
        
    finally:
        shutil.rmtree(jobPath)