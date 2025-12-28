"""
Script to download Indonesian trade data from Google Drive.
Run this locally if you have gdown installed: pip install gdown
"""

import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# List of Google Drive file IDs
file_ids = [
    '1Oyd37Mxp6Z7ojomKM7Q18J-HiuXt7T4e',
    '1lTqCWzDKOVosDOX247BAvxcT1o3qVcUb',
    '1x5sHrqTs79iYrI7VvhWwab6FpBJBUPLB',
    '165jWMAqVnDcovhgLHhxBytPsPg1AKkYN',
    '1I1BrBRVbndcRdZ2DdPO-qgkmvejc2TkX',
    '11ua-kLloQaXIlKFuZlVpTx2oAodTpDmG',
    '1QzF8jCGEblDgot4n5BkSMXDshHcZgGNE',
    '1FmlJK6MfYrDeO0Zoxf2QJMvWzd9kINAl',
    '1m8WTDQkLQSNMhJUti--ZzwfTWVwcFOEi',
    '1MmKzwnBtHm6TT6lgLuQoApItPpN00mHJ'
]

# Download each file
print("Downloading Indonesian trade data files...")
for i, file_id in enumerate(file_ids, 1):
    print(f"\nDownloading file {i}/{len(file_ids)} (ID: {file_id})")
    try:
        import gdown
        url = f'https://drive.google.com/uc?id={file_id}'
        output = f'data/file_{i}.csv'  # Adjust extension as needed
        gdown.download(url, output, quiet=False)
        print(f"✓ Downloaded to {output}")
    except Exception as e:
        print(f"✗ Error downloading file {i}: {e}")

print("\n" + "="*50)
print("Download complete! Check the 'data/' directory")
print("="*50)
