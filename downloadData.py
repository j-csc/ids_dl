import requests
import os

def main():
  print("Starting download...")
  url = "https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/MachineLearningCSV.zip"
  r = requests.get(url)
  with open('./data', "wb") as f:
    f.write(r.content)
  
  # Retrieve HTTP meta-data
  print(r.status_code)
  print(r.headers['content-type'])
  print(r.encoding)

  print("Finished downloading data zip file!")

if __name__ == "__main__":
  main()