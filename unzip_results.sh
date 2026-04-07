#!/bin/bash

# Create the output directory if it doesn't exist
mkdir -p all_results

# Loop through every .tar.gz file inside the zipped_files folder
for f in zipped_results/*.tar.gz; do
    # Get just the filename (e.g., "data.tar.gz")
    base=$(basename "$f")
    
    # Remove the .tar.gz part to get the folder name (e.g., "data")
    foldername="${base%.tar.gz}"
    
    # Define where the files will go
    target_dir="all_results/$foldername"
    
    echo "Unzipping $base into $target_dir..."
    
    # Create the specific folder for this file
    mkdir -p "$target_dir"
    
    # Unzip into that folder, stripping the internal "results/" parent folder
    tar -xvzf "$f" -C "$target_dir" --strip-components=1
done

echo "Done! Check the 'all_results' folder."
