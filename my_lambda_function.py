import json
import boto3
#import datetime
#import time
import uuid
from preprocess import Preprocessor

my_preprocessor = Preprocessor(max_length_tweet=40, max_length_dictionary=100000)
smclient = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    
    tweet = event['tweet']
    
    #start = time.time()
    features = my_preprocessor.preprocess_text(input_text=tweet)
    #end = time.time()
    #pp_time = end-start
    
    model_payload = {
            'features_input':features
            }
    '''model_payload = {
            'request_time':datetime.datetime.now(),
            'feature_input':features,
            'tweet':tweet,
            'sentiment':,
            'probability':,
            'preprocess_time':pp_time,
            'model_inference_time':
            }'''
    #start = time.time()
    response = smclient.invoke_endpoint(
            EndpointName='sentiment-asg5',
            ContentType='application/json',
            Body=json.dumps(model_payload)
            )
    #end = time.time()
    
    #mi_time = end-start
    result={}
    score = json.loads(response['Body'].read().decode())
    if score['predictions'][0][0]>=0.5:
        result['probability']=score['predictions'][0][0]
        result['sentiment']='positive'
    else:
        result['probability']=score['predictions'][0][0]
        result['sentiment']='negative
    
    #result['request_time']=datetime.datetime.now()
    #result['tweet']=tweet
    #result['process_time']=pp_time
    #result['model_inference_time']=mi_time
    print("Result: ", json.dumps(result, indent=2))
    
    '''
    client = boto3.client('s3')
    # Generate a random S3 key name
    upload_key = uuid.uuid4().hex
    client.put_object(Body=json.dumps(result,indent=2), Bucket='ieor4577ziwei/asg5/', Key= upload_key)
    '''
    
    # TODO implement
    return result