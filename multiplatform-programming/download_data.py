#!/usr/bin/env python3
"""
download_triangulations.py

A script to download triangulation files from specified URLs.

Usage:
    python download_triangulations.py

Requirements:
    - Python 3.x
    - requests library (install via `pip install requests`)
"""

import os
import requests
from urllib.parse import urlparse
from tqdm import tqdm

def create_download_directory(directory):
    """
    Creates the download directory if it doesn't exist.
    
    :param directory: Path to the download directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def get_filename_from_url(url):
    """
    Extracts the filename from a URL.
    
    :param url: The URL string.
    :return: Filename as a string.
    """
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)

def download_file(url, save_path):
    """
    Downloads a file from a URL to a specified local path with a progress bar.
    
    :param url: The URL of the file to download.
    :param save_path: The local path where the file will be saved.
    """
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Raise HTTPError for bad responses
            
            # Get the total file size in bytes
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            
            # Initialize progress bar
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=get_filename_from_url(url))
            
            with open(save_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            
            if total_size != 0 and progress_bar.n != total_size:
                print(f"WARNING: Downloaded size for {save_path} does not match expected size.")
            else:
                print(f"Successfully downloaded: {save_path}")
                
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred while downloading {url}: {http_err}")
    except Exception as err:
        print(f"An error occurred while downloading {url}: {err}")

def main():
    # Define the list of triangulation file URLs to download
    triangulation_urls = [
        # Replace these URLs with the actual URLs of the triangulation files you want to download
        "http://www.math.tu-berlin.de/diskregeom/stellar/sample_triangulation1.txt",
        "http://www.math.tu-berlin.de/diskregeom/stellar/sample_triangulation2.txt",
    ]
    
    # Define the directory where files will be downloaded
    download_directory = "downloaded_triangulations"
    
    # Create the download directory if it doesn't exist
    create_download_directory(download_directory)
    
    # Iterate over each URL and download the file
    for url in triangulation_urls:
        filename = get_filename_from_url(url)
        save_path = os.path.join(download_directory, filename)
        
        # Check if the file already exists to avoid re-downloading
        if os.path.exists(save_path):
            print(f"File already exists. Skipping download: {save_path}")
            continue
        
        # Download the file
        download_file(url, save_path)
    
    print("All downloads completed.")

if __name__ == "__main__":
    main()
