# Endpoint Management

Manage AWS SageMaker endpoints.

## Commands

- [`list-endpoints`](#list-endpoints) - List all SageMaker endpoints
- [`delete-endpoint`](#delete-endpoint) - Delete a SageMaker endpoint

---

## list-endpoints

List all SageMaker endpoints in your AWS account.

### Synopsis

```bash
easy_sm list-endpoints [OPTIONS]
```

### Description

The `list-endpoints` command displays all SageMaker endpoints with their status and creation timestamp. Use this to monitor active endpoints and their states.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |

### Examples

#### List all endpoints

```bash
easy_sm list-endpoints
```

Output:
```
production-endpoint  InService  2024-01-15 10:23:45.123000+00:00
staging-endpoint  InService  2024-01-14 14:30:22.456000+00:00
dev-endpoint  Failed  2024-01-13 09:15:33.789000+00:00
test-endpoint  Updating  2024-01-16 11:45:10.012000+00:00
```

#### List endpoints with specific IAM role

```bash
easy_sm list-endpoints -r arn:aws:iam::123456789012:role/CustomRole
```

#### Filter endpoints with grep

```bash
# List only production endpoints
easy_sm list-endpoints | grep production

# List only InService endpoints
easy_sm list-endpoints | grep InService

# Count active endpoints
easy_sm list-endpoints | grep InService | wc -l
```

### Output Format

Each line contains three fields separated by spaces:

```
{endpoint_name}  {status}  {creation_time}
```

**Fields:**
- `endpoint_name`: Name of the endpoint
- `status`: Current endpoint status
- `creation_time`: ISO 8601 timestamp with timezone

### Endpoint Status Values

| Status | Description |
|--------|-------------|
| `Creating` | Endpoint is being created (initial deployment) |
| `InService` | Endpoint is active and serving requests |
| `Updating` | Configuration or model update in progress |
| `SystemUpdating` | SageMaker performing system maintenance |
| `RollingBack` | Update failed, rolling back to previous version |
| `Failed` | Endpoint creation or update failed |
| `Deleting` | Endpoint is being deleted |
| `OutOfService` | Endpoint stopped or unavailable |

### Prerequisites

- AWS credentials configured
- IAM role with SageMaker permissions (`sagemaker:ListEndpoints`)

### Use Cases

#### Monitor endpoint health

```bash
# Check if all endpoints are healthy
easy_sm list-endpoints | grep -v InService

# If output is empty, all endpoints are InService
```

#### Find endpoints to clean up

```bash
# List old failed endpoints
easy_sm list-endpoints | grep Failed

# Delete them
easy_sm list-endpoints | grep Failed | awk '{print $1}' | while read ep; do
  easy_sm delete-endpoint -n $ep
done
```

#### Endpoint inventory

```bash
# Create endpoint inventory report
echo "Endpoint Inventory Report - $(date)" > inventory.txt
easy_sm list-endpoints >> inventory.txt

# Or CSV format
echo "endpoint_name,status,creation_time" > endpoints.csv
easy_sm list-endpoints | awk '{print $1","$2","$3}' >> endpoints.csv
```

### Troubleshooting

**Problem**: No endpoints listed but you have active endpoints.

**Solution**: Check AWS region in config file:

```json
{
    "aws_region": "us-east-1"
}
```

Or specify different region in AWS profile.

**Problem**: "AccessDeniedException"

**Solution**: Add `sagemaker:ListEndpoints` permission to IAM role/user:

```json
{
    "Effect": "Allow",
    "Action": "sagemaker:ListEndpoints",
    "Resource": "*"
}
```

---

## delete-endpoint

Delete a SageMaker endpoint.

### Synopsis

```bash
easy_sm delete-endpoint --endpoint-name NAME [OPTIONS]
```

### Description

The `delete-endpoint` command deletes a SageMaker endpoint, stopping all running instances and removing the endpoint. Optionally, it can also delete the associated endpoint configuration.

!!! warning "Irreversible Action"
    Deleting an endpoint is permanent. The endpoint will stop serving requests immediately. Ensure you have a backup or can redeploy if needed.

### Options

| Option | Short | Type | Required | Default | Description |
|--------|-------|------|----------|---------|-------------|
| `--endpoint-name` | `-n` | string | **Yes** | - | Name of the endpoint to delete |
| `--iam-role-arn` | `-r` | string | No | From `SAGEMAKER_ROLE` | AWS IAM role ARN |
| `--app-name` | `-a` | string | No | Auto-detected | App name for configuration |
| `--delete-config` | - | boolean | No | `false` | Also delete the endpoint configuration |

### Examples

#### Delete an endpoint

```bash
easy_sm delete-endpoint -n my-endpoint
```

Output:
```
my-endpoint
```

#### Delete endpoint and its configuration

```bash
easy_sm delete-endpoint -n my-endpoint --delete-config
```

This deletes both:
- The endpoint: `my-endpoint`
- The endpoint config: `my-endpoint-config`

#### Delete with specific IAM role

```bash
easy_sm delete-endpoint \
  -n old-endpoint \
  -r arn:aws:iam::123456789012:role/CustomRole
```

#### Delete multiple endpoints

```bash
# Delete all failed endpoints
easy_sm list-endpoints | grep Failed | awk '{print $1}' | while read endpoint; do
  echo "Deleting $endpoint..."
  easy_sm delete-endpoint -n $endpoint
done
```

#### Safe deletion with confirmation

```bash
#!/bin/bash
ENDPOINT=$1

echo "About to delete endpoint: $ENDPOINT"
read -p "Are you sure? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
  easy_sm delete-endpoint -n $ENDPOINT --delete-config
  echo "Deleted $ENDPOINT"
else
  echo "Deletion cancelled"
fi
```

### Output Format

The command outputs the deleted endpoint name:

```
my-endpoint
```

### What Gets Deleted

**Without `--delete-config`:**
- ✅ Endpoint instances (stops serving traffic)
- ✅ Endpoint resource
- ❌ Endpoint configuration (remains)
- ❌ Model artifacts in S3 (remain)
- ❌ Docker image in ECR (remains)

**With `--delete-config`:**
- ✅ Endpoint instances
- ✅ Endpoint resource
- ✅ Endpoint configuration
- ❌ Model artifacts in S3 (remain)
- ❌ Docker image in ECR (remains)

!!! tip "Model Artifacts"
    Model artifacts in S3 are never deleted by this command. You can reuse them to redeploy the same endpoint or deploy to a different endpoint.

### Prerequisites

- Active SageMaker endpoint
- IAM role with permissions:
  - `sagemaker:DeleteEndpoint`
  - `sagemaker:DeleteEndpointConfig` (if using `--delete-config`)

### Endpoint Configuration

When you create an endpoint, SageMaker creates an endpoint configuration with name pattern:
```
{endpoint-name}-config
```

This configuration stores:
- Model reference
- Instance type and count
- Data capture configuration
- Production variants

### When to Delete Config

**Delete config when:**
- ✅ Permanently removing the endpoint
- ✅ Won't redeploy with same configuration
- ✅ Cleaning up completely

**Keep config when:**
- ❌ Might redeploy the endpoint soon
- ❌ Want to reuse the configuration
- ❌ Testing temporary changes

### Cost Implications

Deleting an endpoint immediately stops billing for:
- Instance hours
- Data transfer

Model artifacts in S3 continue to incur storage costs (typically very small).

### Redeploying After Deletion

You can redeploy a deleted endpoint using the same model:

```bash
# Delete endpoint
easy_sm delete-endpoint -n my-endpoint --delete-config

# Redeploy later with same model
easy_sm deploy \
  -n my-endpoint \
  -e ml.m5.large \
  -m s3://bucket/models/model.tar.gz
```

### Use Cases

#### 1. Clean Up Development Endpoints

```bash
#!/bin/bash
# cleanup_dev_endpoints.sh

# Delete all dev endpoints older than 7 days
easy_sm list-endpoints | grep "dev-" | while read line; do
  endpoint=$(echo $line | awk '{print $1}')
  created=$(echo $line | awk '{print $3}')

  # Calculate age (simplified)
  echo "Deleting old dev endpoint: $endpoint"
  easy_sm delete-endpoint -n $endpoint --delete-config
done
```

#### 2. Blue-Green Deployment Cleanup

```bash
# After switching traffic from blue to green

# Delete old blue endpoint
easy_sm delete-endpoint -n prod-endpoint-blue --delete-config

# Keep green endpoint running
echo "Green endpoint (prod-endpoint-green) is now serving production traffic"
```

#### 3. Cost Optimization

```bash
# Delete unused endpoints during off-hours

# Weekends: delete non-production endpoints
if [ $(date +%u) -ge 6 ]; then
  easy_sm delete-endpoint -n staging-endpoint
  easy_sm delete-endpoint -n qa-endpoint
fi

# Monday morning: redeploy
# (add to cron or scheduled task)
```

#### 4. Failed Endpoint Cleanup

```bash
# Clean up all failed endpoints
easy_sm list-endpoints | grep Failed | awk '{print $1}' | while read ep; do
  echo "Cleaning up failed endpoint: $ep"
  easy_sm delete-endpoint -n $ep --delete-config
done
```

### Troubleshooting

#### "ResourceNotFound" error

**Problem**: Endpoint doesn't exist.

**Solution**: Verify endpoint name:

```bash
easy_sm list-endpoints | grep my-endpoint
```

Check for typos or wrong AWS region.

#### "Endpoint is updating, cannot delete"

**Problem**: Endpoint currently being updated.

**Solution**: Wait for update to complete:

```bash
# Monitor status
watch -n 10 'easy_sm list-endpoints | grep my-endpoint'

# Delete when status is "InService" or "Failed"
```

Or force delete via AWS CLI:

```bash
aws sagemaker delete-endpoint \
  --endpoint-name my-endpoint \
  --force
```

#### "Cannot delete endpoint config - still in use"

**Problem**: Endpoint config used by other endpoints.

**Solution**: Delete all endpoints using the config first:

```bash
# Find endpoints using this config
aws sagemaker list-endpoints \
  --query "Endpoints[?EndpointName contains 'myapp'].EndpointName"

# Delete them
easy_sm delete-endpoint -n endpoint1
easy_sm delete-endpoint -n endpoint2

# Then delete config
aws sagemaker delete-endpoint-config \
  --endpoint-config-name myapp-config
```

#### Deletion takes long time

**Problem**: Endpoint stuck in "Deleting" status.

**Solution**: This is normal for endpoints with multiple instances or large configurations. Wait 5-10 minutes. If still stuck, contact AWS Support.

### Verification

After deletion, verify:

```bash
# Should not show the deleted endpoint
easy_sm list-endpoints | grep my-endpoint

# Or check via AWS CLI
aws sagemaker describe-endpoint --endpoint-name my-endpoint
# Should return: "Could not find endpoint"
```

### Safety Practices

1. **Always backup model artifacts** before deleting endpoints
2. **Document endpoint configurations** for easy redeployment
3. **Use confirmation scripts** for production deletions
4. **Monitor CloudWatch metrics** before deletion to ensure no active traffic
5. **Keep endpoint configs** if you might redeploy

### Automated Cleanup Script

```bash
#!/bin/bash
# endpoint_cleanup.sh
# Safe deletion with checks

set -e

ENDPOINT=$1

if [ -z "$ENDPOINT" ]; then
  echo "Usage: $0 <endpoint-name>"
  exit 1
fi

# Check if endpoint exists
if ! easy_sm list-endpoints | grep -q "$ENDPOINT"; then
  echo "Error: Endpoint $ENDPOINT not found"
  exit 1
fi

# Check endpoint status
STATUS=$(easy_sm list-endpoints | grep "$ENDPOINT" | awk '{print $2}')

if [ "$STATUS" = "InService" ]; then
  echo "Warning: Endpoint is InService (actively serving traffic)"
  read -p "Are you sure you want to delete? (yes/no): " confirm
  if [ "$confirm" != "yes" ]; then
    echo "Deletion cancelled"
    exit 0
  fi
fi

# Check recent invocations (optional - requires CloudWatch)
echo "Checking recent activity..."
# Add CloudWatch metric check here if needed

# Delete endpoint
echo "Deleting endpoint: $ENDPOINT"
easy_sm delete-endpoint -n "$ENDPOINT" --delete-config

echo "Endpoint deleted successfully"

# Log deletion
echo "$(date): Deleted endpoint $ENDPOINT" >> endpoint_deletions.log
```

Usage:
```bash
chmod +x endpoint_cleanup.sh
./endpoint_cleanup.sh my-endpoint
```

---

## Complete Endpoint Lifecycle

```bash
export SAGEMAKER_ROLE=arn:aws:iam::123456789012:role/SageMakerRole

# 1. Deploy endpoint
easy_sm deploy \
  -n my-endpoint \
  -e ml.m5.large \
  -m s3://bucket/model.tar.gz

# 2. Monitor endpoint
easy_sm list-endpoints | grep my-endpoint

# 3. Use endpoint
# (make predictions via boto3 or API)

# 4. Update endpoint (redeploy with new model)
NEW_MODEL=s3://bucket/new-model.tar.gz
easy_sm deploy \
  -n my-endpoint \
  -e ml.m5.large \
  -m $NEW_MODEL

# 5. Delete when no longer needed
easy_sm delete-endpoint -n my-endpoint --delete-config
```

## Monitoring and Alerting

### CloudWatch Metrics

Monitor endpoint health with CloudWatch:

```bash
# Get invocation count
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name Invocations \
  --dimensions Name=EndpointName,Value=my-endpoint \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Sum

# Get model latency
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=my-endpoint \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average
```

### Alert on Failed Endpoints

```bash
#!/bin/bash
# check_endpoint_health.sh

FAILED=$(easy_sm list-endpoints | grep -c Failed || true)

if [ $FAILED -gt 0 ]; then
  echo "ALERT: $FAILED failed endpoint(s) detected"
  easy_sm list-endpoints | grep Failed

  # Send alert (email, Slack, PagerDuty, etc.)
  # curl -X POST https://hooks.slack.com/... -d "Failed endpoints: $FAILED"

  exit 1
else
  echo "All endpoints healthy"
fi
```

Run periodically via cron:
```cron
*/15 * * * * /path/to/check_endpoint_health.sh
```

## Related Commands

- [`deploy`](deploy.md) - Deploy endpoints
- [`deploy-serverless`](deploy.md#deploy-serverless) - Deploy serverless endpoints
- [`train`](train.md) - Train models for deployment

## See Also

- [Endpoint Management Guide](../user-guide/endpoint-management.md)
- [Monitoring and Logging](../user-guide/monitoring.md)
- [SageMaker Endpoints Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
