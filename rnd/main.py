import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError


def validate_s3_connection(endpoint_url, access_key, secret_key, region_name="us-east-1"):
    try:
        # Create S3 client
        s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            # For custom endpoints like MinIO, Wasabi, etc.
            endpoint_url=endpoint_url,
            region_name=region_name,
        )

        # Try listing buckets (basic connectivity check)
        response = s3.list_buckets()
        print("✅ Connection successful!")
        print("Buckets:", [b["Name"] for b in response.get("Buckets", [])])

    except NoCredentialsError:
        print("❌ No credentials found. Please check your access key and secret key.")
    except PartialCredentialsError:
        print("❌ Incomplete credentials provided.")
    except ClientError as e:
        print(f"❌ Client error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


# Example usage
validate_s3_connection(
    endpoint_url="https://qykqwcvbcvcteudugdwm.storage.supabase.co/storage/v1/s3",
    access_key="323344acf357c1104aaca90178c8be1f",
    secret_key="fa61216ab65ad43f6e951cf67a979690344b6e75f785a5a87b86ec2995abc164"
)
