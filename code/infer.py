import os
import shutil
import copy
import numpy as np

import kafkajobs

from model import detect_head_and_body

def infer_in_json_field(model, images_json_field):
    '''Takes serialized images and applied detection model for them'''

    imagesNp = kafkajobs.serialization.imagesFieldToNp(images_json_field)
    heads = list()
    annotations = list()
    heads_counts = list()
    for imNumpy in imagesNp:
        rank = len(imNumpy.shape)
        #print(f"shape {imNumpy.shape}")
        if rank == 2:
            # grayscale image
            imNumpy = np.stack((imNumpy, imNumpy, imNumpy), axis=2)
            print("image of rank 2. grayscale image converted to 3 channels")
        elif rank == 3:
            # color image
            colChannels = imNumpy.shape[2]
            if colChannels == 2:
                # grayscale image + alpha
                # we skip such for now. Most probably it is PNG with clipart
                print("2 color channels detected, skipping such image")
                continue
            elif colChannels > 3:
                imNumpy = imNumpy[:,:,:3] # remove alpha channel if any
                print("Image has {0} channels, but we only support 3. alpha channel removed".format(colChannels))
        elif rank == 4:
            # multiframe gif?
            imNumpy = imNumpy[0,:,:,:] # take first frame
            print("multiframe gif detected, taking first frame")
        
        w,h = imNumpy.shape[1], imNumpy.shape[0]
        if w<5 or h<5:
            print("image too small, skipping. Shape is {0}".format(imNumpy.shape))
            continue

        #print(f"shape is {imNumpy.shape}")
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