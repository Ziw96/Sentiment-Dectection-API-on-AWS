# AI-OR_in_the_cloud_asg5
Assignment 5 for IEOR4577

Please see below for instructions:
- IEOR4577 Homework5.docx: detailed steps and report
- preprocess-new.zip: full package for developing the model lambda function
- my_lambda_function.py: main code for calling preprocess and model modules
- preprocess.py: preprocess file
- dict: folder containing the essential word dictionary to run code
- requirement.txt: package requirements
- Others: files for running code and sample json log output


API link: https://cevxerp424.execute-api.us-west-2.amazonaws.com/v1 

Call API method:
curl -X POST https://cevxerp424.execute-api.us-west-2.amazonaws.com/v1/predict --header "Content-Type:application/json" --data '{"tweet": "I love apples"}'

Please contact ziwei.li@columbia.edu if you would like to test the API

Sample Payload log info: https://sentimentlog.s3-us-west-2.amazonaws.com/015ca6008c0b4c13b929067b1e13ce68
