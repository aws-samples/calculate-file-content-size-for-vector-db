# Calculate file content size for vector db

### Why use this tool?
TL;DR - To calculate directionally accurate data size (and hence costs) for vector databases.

File systems report PDF file sizes 5x to 20x the size of the actual text content in it because of the overhead associated with formatting and pictures in it. Using the file system size, if you were to estimate the vector database size and costs, you will over-estimate the costs by more than 5x.

Thus, you need a tool that reads through the PDF files and estimates the actual text in it to give you a better estimate of the text content size that will be be chunked, vectorized, and stored in vector database.

## Sample output of the tool.
After scanning all the files, the tool outputs a summary as shown below:
  ```
The total file size is 441,354,647 and text size is 23,765,682 with a ratio of 18.57 (after processing 34 files). You can apply the ratio to estimate the text size for rest of your files. For example if the file system size reports 100 GB, applying the ratio 18.57 yields a text size of 5.38 GB. You can use this size to estimate vector database size.
  ```
<br>The tool also prints the results in a tabular format as shown below and stores it in a CSV file.
```
+--------------------------+------------+---------+----------+-------------+-------------+---------+
| Filename                 |   Time (s) | Pages   | Chunks   | File Size   | Text Size   |   Ratio |
+==========================+============+=========+==========+=============+=============+=========+
| bedrock-meetups.pdf      |      21.56 | 1652    | 2921     | 12,318,934  | 2,427,921   |    5.07 |
+--------------------------+------------+---------+----------+-------------+-------------+---------+
| bedrock-features.pdf     |       0.27 | 15      | 25       | 463,254     | 20,982      |   22.08 |
+--------------------------+------------+---------+----------+-------------+-------------+---------+
| Bedrock Models.pdf       |       0.22 | 24      | 32       | 314,167     | 21,298      |   14.75 |
+--------------------------+------------+---------+----------+-------------+-------------+---------+
| SUM TOTAL                |      22.63 | 1,691   | 2,978    | 13,096,355  | 2,470,201   |    5.3  |
+--------------------------+------------+---------+----------+-------------+-------------+---------+
``` 
<br><br><b>Tip:</b> Run this tool for a sample representation of files to identify the ratio. Then apply this ratio across all your files to estimate the actual text size across all the files. 
<br>For example, if you run this tool against a sample set of 20 files and find the ratio to be 10. Assume that you have 100s of PDF files for which the file system reports the total size as 50 GB. To identify the approximate text size in these files, you will use the formula: 
<br>Size of text content = (file system size) / (ratio). In this example, total size of text = 5 GB.

### Features
- Processes PDF files in one or more directories
- Calculates file size, text size, and size ratio for each file and a summary for all PDFs
- Stores the output in a CSV file. It creates one CSV file for each directory path you specify.
- Supports multi-threaded processing for improved performance
- Smart processing ensures that large files are sorted first to keep the overall processing window small

### How to use the tool
Run the script from the command line using the following format:
  ```
python pdf_reader.py <folder_path 1> <folder path 2> <number_of_threads>
  ```

### Arguments

1. `folder_path`: The path to the folder containing the PDF files you want to process. You can specify more than one folder separated by spaces. If your folder name has spaces, please specify it in "". 
2. `number_of_threads`: The number of threads to use for processing. Adjust this based on your system's capabilities. Specify this as the last parameter.

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/aws-samples/calculate-file-content-size-for-vector-db.git
cd calculate-file-content-size-for-vector-db
```

2. Create and activate a virtual environment:

- On Windows:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- On macOS and Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

3. Install the required packages:
  ```bash
  pip install -r requirements.txt
  ```

### Usage Example

After setting up the environment, use the tool. Example usage is below.

  ```bash
python pdf_reader.py /path/to/pdf/folder 4
python pdf_reader.py /path/to/pdf/folder1 /another/path/folder2 4
  ```
In the above command, 4 is the number of threads.

### Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment
```
deactivate
```

### Troubleshooting

If you encounter any issues:
- Run the following command to learn how to use the tool correctly.

```bash
python pdf_reader.py -h
```
- Ensure you have the correct Python version installed.
- Make sure you've activated the virtual environment before running the script.
- Check that all required packages are installed correctly.

## Roadmap

- Support html files, word documents 
- Allow an S3 folder to be specified

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.