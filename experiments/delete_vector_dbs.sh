#!/bin/bash

# Function to handle errors
handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting vector database cleanup..."

# Get list of vector databases
db_list=$(llama-stack-client vector_dbs list) || handle_error "Failed to get vector database list"

# Extract database identifiers using awk
# Look for lines starting with │ and extract the first column
db_ids=$(echo "$db_list" | awk -F'│' '/^│/ && NF>1 {gsub(/^[[:space:]]+|[[:space:]]+$/,"",$2); if($2 != "identifier" && $2 != "") print $2}')

# Initialize counters
total_dbs=$(echo "$db_ids" | wc -l)
success_count=0
failed_count=0

echo "Found $total_dbs vector databases to unregister"
echo "----------------------------------------"

# Process each database
while IFS= read -r db_id; do
    echo "Unregistering: $db_id"
    if llama-stack-client vector_dbs unregister "$db_id"; then
        echo -e "${GREEN}✓ Successfully unregistered: $db_id${NC}"
        ((success_count++))
    else
        echo -e "${RED}✗ Failed to unregister: $db_id${NC}"
        ((failed_count++))
    fi
    echo "----------------------------------------"
done <<< "$db_ids"

# Print summary
echo "Cleanup complete!"
echo "Summary:"
echo -e "${GREEN}- Successfully unregistered: $success_count${NC}"
echo -e "${RED}- Failed to unregister: $failed_count${NC}"
echo "- Total processed: $total_dbs"