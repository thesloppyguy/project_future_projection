import boto3

s3 = boto3.client(
    "s3",
    aws_secret_access_key='762b959669ab2e6419a5b43aee5a877f00ca93ed388783d6250cd4dcd127507f',
    aws_access_key_id='3bf5411cd14bc68594ecbc82576ab1dc',
    endpoint_url='https://qykqwcvbcvcteudugdwm.storage.supabase.co/storage/v1/s3',
    region_name='ap-southeast-1',
)
response = s3.list_buckets()

print(response)
