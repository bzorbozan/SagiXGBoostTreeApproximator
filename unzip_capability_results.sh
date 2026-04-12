# need to give the script permission to execute by running: chmod +x unzip_capability_results.sh
# ./unzip_capability_results.sh to run script
# ____________________________________________________________

#!/bin/bash

# Folder containing the .tar.gz files
SOURCE_PARENT="zipped_capability_results"

# Folder where you want the unzipped results to go
TARGET_PARENT="capability_results"

# The specific files to process
FILES=("fbt_capability_results.tar.gz" "shapefbt_capability_results.tar.gz")

# Loop through each file
for FILE_NAME in "${FILES[@]}"; do
    FILE_PATH="$SOURCE_PARENT/$FILE_NAME"

    if [[ -f "$FILE_PATH" ]]; then
        # Create folder name from filename (e.g., fbt_capability_results)
        DIR_NAME="${FILE_NAME%.tar.gz}"
        
        # Combine to create the final destination path
        # Example: capability_results/fbt_capability_results
        DESTINATION_PATH="$TARGET_PARENT/$DIR_NAME"

        echo "------------------------------------------"
        echo "Found: $FILE_PATH"
        echo "Extracting to: $DESTINATION_PATH"

        # Create the destination directory (and its parent if it doesn't exist)
        mkdir -p "$DESTINATION_PATH"

        # Extract the file
        # -C ensures it extracts into the NEW capability_results folder
        tar -xzf "$FILE_PATH" -C "$DESTINATION_PATH" --strip-components=1

        if [ $? -eq 0 ]; then
            echo "Success: $FILE_NAME processed."
        else
            echo "Error: Failed to extract $FILE_NAME"
        fi
    else
        echo "Warning: $FILE_PATH not found, skipping..."
    fi
done

echo "------------------------------------------"
echo "All tasks complete. Check the '$TARGET_PARENT' folder."