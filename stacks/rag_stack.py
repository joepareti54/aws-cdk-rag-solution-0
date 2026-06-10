import aws_cdk.aws_s3_notifications as s3n
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_apigateway as apigw,
)
from constructs import Construct

class RAGStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 Bucket for documents and FAISS index
        self.bucket = s3.Bucket(
            self, "RAGBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # Lambda Layer for shared dependencies
        dependencies_layer = lambda_.LayerVersion(
            self, "DependenciesLayer",
            code=lambda_.Code.from_asset("layers/dependencies"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_9],
            description="FAISS, numpy, and other dependencies",
        )

        # Common environment variables
        common_env = {
            "BUCKET_NAME": self.bucket.bucket_name,
            "EMBEDDING_MODEL_ID": "amazon.titan-embed-text-v2:0",
#            "LLM_MODEL_ID": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "LLM_MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "INDEX_PREFIX": "faiss-index/",
        }

        # IAM Policy for Bedrock
        bedrock_policy = iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
            ],
            resources=["*"],
        )

        # Document Processor Lambda
        self.document_processor = lambda_.Function(
            self, "DocumentProcessor",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="document_processor.handler",
            code=lambda_.Code.from_asset("lambdas/document_processor"),
            timeout=Duration.minutes(5),
            memory_size=1024,
            environment=common_env,
            layers=[dependencies_layer],
        )
        self.document_processor.add_to_role_policy(bedrock_policy)
        self.bucket.grant_read_write(self.document_processor)
        # Trigger Document Processor when files are uploaded
#        self.bucket.add_event_notification(s3.EventType.OBJECT_CREATED,s3n.LambdaDestination(self.document_processor),s3n.LambdaDestination(self.document_processor),s3.NotificationKeyFilter(prefix="uploads/"))
        self.bucket.add_event_notification(s3.EventType.OBJECT_CREATED, s3n.LambdaDestination(self.document_processor), s3.NotificationKeyFilter(prefix="uploads/"))

        # Index Manager Lambda
        self.index_manager = lambda_.Function(
            self, "IndexManager",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="index_manager.handler",
            code=lambda_.Code.from_asset("lambdas/index_manager"),
            timeout=Duration.minutes(10),
            memory_size=3008,
            environment=common_env,
            layers=[dependencies_layer],
        )
        self.index_manager.add_to_role_policy(bedrock_policy)
        self.bucket.grant_read_write(self.index_manager)

        # Query Handler Lambda
        self.query_handler = lambda_.Function(
            self, "QueryHandler",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="query_handler.handler",
            code=lambda_.Code.from_asset("lambdas/query_handler"),
            timeout=Duration.seconds(30),
            memory_size=3008,
            environment=common_env,
            layers=[dependencies_layer],
#            reserved_concurrent_executions=2,
        )
        self.query_handler.add_to_role_policy(bedrock_policy)
        self.bucket.grant_read(self.query_handler)

        # API Gateway
        api = apigw.RestApi(
            self, "RAGAPI",
            rest_api_name="RAG Service",
            description="RAG Query API",
            deploy_options=apigw.StageOptions(stage_name="prod"),
        )

        query_resource = api.root.add_resource("query")
        query_resource.add_method(
            "POST",
            apigw.LambdaIntegration(self.query_handler),
            api_key_required=True,
        )
        api_key = api.add_api_key("RAGApiKey", api_key_name="rag-default-key", description="Default key for RAG API access",)
        usage_plan = api.add_usage_plan("RAGUsagePlan",name="rag-default-plan",throttle=apigw.ThrottleSettings(rate_limit=2,burst_limit=5,),quota=apigw.QuotaSettings(limit=1000,period=apigw.Period.DAY,))

        usage_plan.add_api_key(api_key)
        usage_plan.add_api_stage(stage=api.deployment_stage)
        # Outputs
        CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
        CfnOutput(self, "APIEndpoint", value=api.url)
        CfnOutput(self, "DocumentProcessorArn", value=self.document_processor.function_arn)
        CfnOutput(self, "IndexManagerArn", value=self.index_manager.function_arn)
        CfnOutput(self, "ApiKeyId", value=api_key.key_id, description="Use 'aws apigateway get-api-key --api-key <id> --include-value' to retrieve",)
