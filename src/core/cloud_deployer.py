import boto3


def deploy_to_aws_lambda(function_name, zip_file):
    """""\"
Summary of deploy_to_aws_lambda.

Parameters
----------
function_name : type
    Description of parameter `function_name`.
zip_file : type
    Description of parameter `zip_file`.

Returns
-------
None
""\""""
    client = boto3.client('lambda')
    with open(zip_file, 'rb') as f:
        zipped_code = f.read()
    response = client.update_function_code(FunctionName=function_name,
        ZipFile=zipped_code)
    return response
