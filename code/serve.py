import base64
import shutil
import os
import sys
import io

from PIL import Image as im

import torch

import kafkajobs
from model import detect_head_and_body
from infer import infer_in_json_field

kafkaUrl = os.environ['KAFKA_URL']
inputQueueName = os.environ['INPUT_QUEUE']
outputQueueName = os.environ['OUTPUT_QUEUE']

appName = "zhiru-calvin-yolo5-head-and-body-detector"

worker = kafkajobs.jobqueue.JobQueueWorker(appName, kafkaBootstrapUrl=kafkaUrl, topicName=inputQueueName, appName=appName)
resultQueue = kafkajobs.jobqueue.JobQueueProducer(kafkaUrl, outputQueueName, appName)

workdir = '/tmp'

sys.path.append("./yolov5/")
model = torch.hub.load('./yolov5/', 'custom', path='yolov5s.pt', source='local')
print("model loaded")

def work():
    print("Service started. Pooling for a job")
    while True:        
        job = worker.GetNextJob(5000)
        #print("Got job {0}".format(job))
        uid = job["uid"]
        print("{0}: Starting to process the job".format(uid))
        images = job['images']
        print("{0}: Extracting {1} images".format(uid, len(images)))

        jobPath = os.path.join(workdir,uid)
        os.mkdir(jobPath)

        try:
            yolo5_outputs = infer_in_json_field(model, images)

            job['yolo5_output'] = yolo5_outputs

            resultQueue.Enqueue(uid, job)
            worker.Commit()
            print("{0}: Job processed successfully, results are submited to kafka".format(uid))
        finally:
            shutil.rmtree(jobPath)

work()