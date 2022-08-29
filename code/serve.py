import os
import sys

import torch

import kafkajobs
from infer import process_job

kafkaUrl = os.environ['KAFKA_URL']
inputQueueName = os.environ['INPUT_QUEUE']
outputQueueName = os.environ['OUTPUT_QUEUE']

appName = "zhiru-calvin-yolo5-head-and-body-detector"

worker = kafkajobs.jobqueue.JobQueueWorker(appName, kafkaBootstrapUrl=kafkaUrl, topicName=inputQueueName, appName=appName)
resultQueue = kafkajobs.jobqueue.JobQueueProducer(kafkaUrl, outputQueueName, appName)

sys.path.append("./yolov5/")
model = torch.hub.load('./yolov5/', 'custom', path='yolov5s.pt', source='local')
print("model loaded")

def work():
    print("Service started. Pooling for a job")
    while True:        
        job = worker.GetNextJob(5000)
        
        print("Got job {0}".format(job["uid"]))

        out_job, uid = process_job(model, job)
        
        resultQueue.Enqueue(uid, out_job)
        worker.Commit()
        print("{0}: Job processed successfully, results are submited to kafka".format(uid))

work()