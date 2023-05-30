#!/bin/bash

# Check if a limit argument is provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <limit> [--verbose]"
  exit 1
fi

# Set the limit for the number of zip files to download
limit=$1

# Check for the verbose flag
verbose=0
if [ "$#" -gt 1 ] && [ "$2" == "--verbose" ]; then
  verbose=1
fi

# Set the base URL you want to download the files from
base_url="https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/"

# Set the target directory for extracting the zip files
target_dir="/home/toomuch/kaggle-diffusion/private-cv/test-images"

# Create the target directory if it doesn't exist
mkdir -p "${target_dir}"

# Initialize progress bar variables
width=$(tput cols)
progress_chars=$(($width - 25))

# Download and extract the zip files with the specified pattern
for ((i = 1; i <= limit; i++)); do
  # Generate the zip file URL with the counter
  zip_url="${base_url}part-$(printf "%06d" $i).zip"
  zip_filename="$(basename ${zip_url})"
  
  # Download the zip file with or without wget output
  if [ $verbose -eq 1 ]; then
    wget -nc "${zip_url}"
  else
    wget -qnc "${zip_url}"
  fi
  
  # Extract the content of the zip file to the target directory with or without unzip output
  if [ $verbose -eq 1 ]; then
    unzip -n "${zip_filename}" -d "${target_dir}"
  else
    unzip -qn "${zip_filename}" -d "${target_dir}"
  fi
  
  # Remove the downloaded .zip file
  rm "${zip_filename}"
  
  # Update the progress bar
  printf "\r[%-*s] %3d%%" $progress_chars "$(seq -s= $(($i * $progress_chars / $limit)) | tr -d '[:digit:]')" $(($i * 100 / $limit))
done

# Print newline after the progress bar
echo