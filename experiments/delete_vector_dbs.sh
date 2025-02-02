#!/bin/bash

# Array of vector database IDs to unregister
declare -a vector_dbs=(
    "vector_db_004249ce-3edc-4c30-b290-27458d075ebf"
    "vector_db_0b910398-2d02-4961-9a7c-84b1b07050cc"
    "vector_db_13490f13-9429-49e8-8d1f-6774af723cde"
    "vector_db_20c100af-15a5-4be1-b0fd-923bf9789109"
    "vector_db_3ed181ad-c366-4458-bee5-8c9374793631"
    "vector_db_50f3af23-bc50-4ff2-a833-ba8013d7bc2c"
    "vector_db_559742b6-b91b-4880-a96f-1ce80b1afa7a"
    "vector_db_a1590c07-d402-4bcb-976c-832b8eb8d22f"
    "vector_db_e7783f48-c333-4a5e-8db1-a948ec414814"
    "vector_db_e8303fcd-35b5-404b-a71c-a30417a388a9"
    "vector_db_f0104d98-ff55-4bc1-8183-0d87097fb249"
)

# Counter for successful and failed operations
success_count=0
failed_count=0

# Print start message
echo "Starting vector database unregistration process..."
echo "Total databases to process: ${#vector_dbs[@]}"
echo "----------------------------------------"

# Process each vector database
for db_id in "${vector_dbs[@]}"; do
    echo "Processing: $db_id"
    
    if llama-stack-client vector_dbs unregister "$db_id"; then
        echo "✓ Successfully unregistered: $db_id"
        ((success_count++))
    else
        echo "✗ Failed to unregister: $db_id"
        ((failed_count++))
    fi
    echo "----------------------------------------"
done

# Print summary
echo "Unregistration process complete!"
echo "Summary:"
echo "- Successfully unregistered: $success_count"
echo "- Failed to unregister: $failed_count"
echo "- Total processed: ${#vector_dbs[@]}"
