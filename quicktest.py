from app.triage import TriagePipeline

pipeline = TriagePipeline()  # reads .env, builds KB index from data/knowledge-base
result = pipeline.run({
    "subject": "How do I configure Exports in AnalyticsHub?",
    "body": "Hi,\n\nI'm trying to set up Exports for our team but can't find clear documentation. Specifically, I need to know:\n\n1. How to assign to team\n2. What permissions are required\n3. Whether this integrates with Snowflake\n\nWe're on the Starter plan. Thanks in advance.",
})
print(result.model_dump_json(indent=2))