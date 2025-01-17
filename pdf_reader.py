import json
import multiprocessing
import os
import re
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def _fmt_num(value):
    """
    Formats a numeric value with thousands separators and no decimal places.
    Example: _fmt_num(1234567.89) returns "1,234,567".
    """
    return "{:,.0f}".format(value)


def _fmt_num_2(value, decimals=0):
    """
    Formats a numeric value with thousands separators and the specified number of decimals.
    If decimals is not provided, it defaults to 0.
    Example: _fmt_num_2(1234567.89, 2) returns "1,234,567.89".
    """
    format_str = f"{{:,.{decimals}f}}"
    return format_str.format(value)


import pandas as pd
from tabulate import tabulate


def _to_table_format(results):
    """
    Convert processing results into a formatted table and DataFrame.

    This function takes a list of processing results and converts them into three formats:
    1. A list of headers for a table
    2. A list of lists containing the data for each row of the table
    3. A pandas DataFrame containing all the data

    Args:
    results (list): A list of dictionaries, each containing the results of processing a file.
                    Each dictionary should have keys: 'file_path', 'seconds', 'pages',
                    'chunk_list', 'file_size', 'text_size', and 'ratio'.

    Returns:
    tuple: A tuple containing:
        - headers (list): Column names for the table
        - table_data (list): A list of lists, each inner list representing a row of data
        - df (pandas.DataFrame): A DataFrame containing all the processed data

    Note:
    - File sizes are formatted with commas as thousand separators
    - Time and ratio are formatted to two decimal places
    - The 'Chunks' column represents the length of the 'chunk_list' for each file
    """
    # Define the headers for the table
    headers = [
        "Filename",
        "Time (s)",
        "Pages",
        "Chunks",
        "File Size",
        "Text Size",
        "Ratio",
    ]
    table_data = []
    rows_list = []

    for result in results:
        # Extract filename from the full path
        filename = os.path.basename(result["file_path"])

        # Format the data for display
        time = f"{result['seconds']:.2f}"
        pages = str(result["pages"])
        chunks = str(len(result["chunk_list"]))
        file_size = f"{result['file_size']:,}"
        text_size = f"{result['text_size']:,}"
        ratio = f"{result['ratio']:.2f}"

        # Append formatted data to table_data for display purposes
        table_data.append([filename, time, pages, chunks, file_size, text_size, ratio])

        # Append data to rows_list for DataFrame creation
        rows_list.append(
            {
                "Filename": filename,
                "Time (s)": time,
                "Pages": pages,
                "Chunks": chunks,
                "File Size": file_size,
                "Text Size": text_size,
                "Ratio": ratio,
            }
        )

    # Create a DataFrame from the processed data
    df = pd.DataFrame(rows_list)
    return headers, table_data, df


def _print_and_save_results(results, total_time, output_file, print_metadata):
    """
    Print and save processing results for multiple files.

    This function processes a list of results from file processing, calculates totals,
    prints a summary table, and saves the results to a CSV file.

    Args:
    results (list): A list of dictionaries, each containing the results of processing a file.
                    Each dictionary should have keys: "file_size", "text_size", "pages", "chunk_list".
    total_time (float): The total processing time for all files.
    output_file (str): The path to the CSV file where results will be saved.
    print_metadata (bool): If True, prints detailed metadata for each file.

    Returns:
    None

    Side effects:
    - Prints a summary table to the console.
    - Saves results to a CSV file.
    - Prints additional statistics and estimates.

    Note:
    This function uses helper functions _fmt_num, _fmt_num_2, and _to_table_format,
    which should be defined elsewhere in the code.
    """
    # Initialize counters
    n_files = 0
    total_file_size = total_text_size = total_pages = total_chunks = 0

    # Process each result
    for result in results:
        metadata = json.dumps(result, indent=4)
        if print_metadata:
            print(metadata)

        # Accumulate totals
        total_file_size += result.get("file_size", "0")
        total_text_size += result.get("text_size", "0")
        total_pages += result.get("pages", 0)
        total_chunks += len(result.get("chunk_list", []))
        n_files += 1

    # Calculate overall ratio
    total_ratio = total_file_size / total_text_size

    # Generate table data
    headers, table_data, df = _to_table_format(results)

    # Add summary row to table data
    table_data.append(
        [
            "SUM TOTAL",
            _fmt_num_2(total_time, 2),
            _fmt_num(total_pages),
            _fmt_num(total_chunks),
            _fmt_num(total_file_size),
            _fmt_num(total_text_size),
            _fmt_num_2(total_ratio, 2),
        ]
    )

    # Print summary table
    table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    print(
        f"\n***********Following is the summary for {n_files} files****************\n"
    )
    print(table_str)

    # Add summary row to DataFrame and save to CSV
    new_row = pd.DataFrame(
        {
            "Filename": ["SUM TOTAL"],
            "Time (s)": [_fmt_num_2(total_time, 2)],
            "Pages": [_fmt_num(total_pages)],
            "Chunks": [_fmt_num(total_chunks)],
            "File Size": [_fmt_num(total_file_size)],
            "Text Size": [_fmt_num(total_text_size)],
            "Ratio": [_fmt_num_2(total_ratio, 2)],
        }
    )
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(output_file, index=False)

    # Print additional statistics and estimates
    print(f"The above table is also stored in CSV file: {output_file}")
    print(
        f"The total file size is {_fmt_num(total_file_size)} and text size is {_fmt_num(total_text_size)} with a ratio of {_fmt_num_2(total_ratio, 2)} (after processing {n_files} files). You can apply the ratio to estimate the text size for rest of your files. For example if the file system size reports 100 GB, applying the ratio {_fmt_num_2(total_ratio, 2)} yields a text size of {_fmt_num_2(100/total_ratio, 2)} GB. You can use this size to estimate vector database size."
    )


def _folder_path_to_filename(folder_path):
    """
    Convert a folder path to a valid file name.

    Args:
    folder_path (str): The path to the folder.

    Returns:
    str: A sanitized file name based on the folder name.
    """
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    sanitized_name = re.sub(r"[^\w\-]", "_", folder_path)

    # Ensure the name doesn't start with a hyphen or underscore
    sanitized_name = sanitized_name.lstrip("-_")

    # If the name is empty after sanitization, use a default name
    if not sanitized_name:
        sanitized_name = "folder"

    return sanitized_name


def _parse_arguments():
    """
    Parse and validate command line arguments for PDF processing script.

    This function sets up the argument parser, defines the expected arguments,
    and performs validation on the provided inputs.

    Returns:
    tuple: A tuple containing:
        - folder_paths (list): List of validated folder paths containing PDF files
        - number_of_threads (int): Number of threads to use for processing
        - print_metadata (bool): Flag indicating whether to print detailed metadata

    Raises:
    SystemExit: If invalid arguments are provided (via argparse.ArgumentParser.error())

    Example usage:
        folder_paths, num_threads, print_metadata = _parse_arguments()

    Note:
    - The last argument in the command line is always interpreted as the thread count.
    - At least one folder path and the thread count must be provided.

    """
    # Set up the argument parser with description and usage examples
    parser = argparse.ArgumentParser(
        description="Process PDF files in one or more folders using multiple threads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python script.py /path/to/pdf/folder 4
        python script.py /path/to/pdf/folder1 /another/path/folder2 4
        python script.py /path/to/pdf/folder 8 --print_metadata

        Note:
        - The folder paths should be an existing directory containing PDF files.
        - The number of threads should be a positive integer.
        - Use --print_metadata to display detailed information about each processed file.
                """,
    )

    # Define command line arguments
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more folder paths containing PDF files. The last argument must be thread count.",
    )

    parser.add_argument(
        "--print_metadata",
        action="store_true",
        help="Print detailed metadata for each file",
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Separate folder paths and number of threads
    folder_paths = args.paths[:-1]
    try:
        number_of_threads = int(args.paths[-1])
    except ValueError:
        parser.error("The last argument must be a number (number of threads)")

    # Validate folder paths
    for path in folder_paths:
        if not os.path.isdir(path):
            parser.error(
                f"The specified folder path '{path}' does not exist or is not a directory."
            )

    # Validate number of threads
    if number_of_threads <= 0:
        parser.error("The number of threads must be a positive integer.")

    return folder_paths, number_of_threads, args.print_metadata


class File_Reader:
    def __init__(self):
        """
        Empty constructor for File_Reader class.
        """
        self.dir_path = None
        self.n_threads = None
        self.file_type = None
        self.chunk_size = None
        self.chunk_overlap = None
        self.files = []
        self.df = None

    def _init(
        self, dir_path, n_threads, file_type=".pdf", chunk_size=1200, chunk_overlap=0
    ):
        """
        Initializes the File_Reader with directory path, number of threads, and file type.

        Args:
        dir_path (str): Path to the directory containing files to process.
        n_threads (int): Number of threads to use for processing.
        file_type (str): File extension to filter (e.g., '.pdf', '.txt').
        """

        self.dir_path = dir_path
        self.n_threads = n_threads
        self.file_type = file_type.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_files(self):
        """
        Helper method to get all files of the specified type in the directory.
        """
        # Ensure the extension starts with a dot
        if not self.file_type.startswith("."):
            self.file_type = "." + self.file_type

        # List to store matching files
        matching_files = []

        try:
            # Walk through the directory and its subdirectories
            for root, dirs, files in os.walk(self.dir_path):
                for filename in files:
                    # Check if the file has the specified extension
                    if filename.lower().endswith(self.file_type.lower()):
                        full_path = os.path.join(root, filename)
                        matching_files.append(full_path)

            return matching_files

        except FileNotFoundError:
            print(f"Error: The directory '{self.dir_path}' does not exist.")
            return []
        except PermissionError:
            print(
                f"Error: Permission denied to access the directory '{self.dir_path}'."
            )
            return []
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return []

    def _sort_files_by_size(self, file_list):
        """
        Sort files by size in decreasing order. This is useful because the large files will start first in a multi-threaded application.
        This allows them to run earlier while smaller files and medium ones come and go. This avoids the scenario where the larger files start late and take up time.

        Args:
        file_list (list): A list of file paths.

        Returns:
        list: A list of file paths sorted by size in decreasing order.
        """
        # Create a list of tuples (file_path, file_size)
        file_size_list = []
        for file_path in file_list:
            try:
                # Get the file size
                file_size = os.path.getsize(file_path)
                file_size_list.append((file_path, file_size))
            except OSError as e:
                print(f"Error accessing file {file_path}: {e}")

        # Sort the list based on file size in descending order
        sorted_files = sorted(file_size_list, key=lambda x: x[1], reverse=True)

        # Extract just the file paths from the sorted list
        sorted_file_paths = [file_path for file_path, _ in sorted_files]

        return sorted_file_paths

    def _preprocess_pdf_content(self, content):
        """
        The preprocess_pdf_content function takes a string content as input and performs several operations to preprocess the content.
        """
        # Replace multiple newline characters (\n) with a single newline
        content = re.sub(r"\n{2,}", "\n", content)
        # Replace newline characters (\n) with a space ( )
        content = re.sub(r"\n{1,}", " ", content)
        # Remove Unicode characters represented as \uXXXX (where X is a hexadecimal digit)
        content = re.sub(r"\\u[0-9a-fA-F]{4}", "", content)
        # Convert the entire string to lowercase
        content = content.lower()

        return content

    def _read_pdf_file(
        self, file_path, chunk_size, chunk_overlap_size, b_get_content=False
    ):
        """
        Read a PDF file, split it into chunks, and process its content.

        This method reads a PDF file, splits its content into overlapping chunks,
        and processes each chunk. It calculates various metrics about the file
        and its content.

        Args:
            file_path (str): Path to the PDF file to be processed.
            chunk_size (int): The size of each text chunk in characters.
            chunk_overlap_size (int): The size of the overlap between text chunks in characters.
            b_get_content (bool, optional): If True, include the chunk content in the output. Defaults to False.

        Returns:
            dict: A dictionary containing processed file information:
                - file_path (str): Path to the processed file
                - seconds (float): Time taken to process the file
                - pages (int): Number of pages in the PDF
                - chunk_list (list): List of dictionaries, each representing a chunk with metadata
                - file_size (int): Size of the file in bytes
                - text_size (int): Total size of extracted text in characters
                - ratio (float): Ratio of file size to text size

        Raises:
            Any exceptions raised by PyPDFLoader or RecursiveCharacterTextSplitter

        Note:
            This method assumes the existence of a _preprocess_pdf_content method in the class.
        """
        # Initialize variables
        documents = []
        s_time = time.time()

        # Load the PDF content
        loader = PyPDFLoader(file_path=file_path)
        documents += loader.load()

        # Initialize text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap_size,
            add_start_index=True,
        )

        # Split documents into chunks
        chunks = text_splitter.split_documents(documents)
        chunk_list = []
        page_number = 0  # page number in the file
        chunk_offset_in_file = 0  # chunk offset in the file

        # Process each chunk
        for ind, chunk in enumerate(chunks):
            content = self._preprocess_pdf_content(
                chunk.page_content
            )  # Preprocess chunk content
            if not b_get_content:
                chunk_content = ""
            else:
                chunk_content = content
            page_number = chunk.metadata["page"]  # this is 0 based.
            chunk_offset_in_page = chunk.metadata[
                "start_index"
            ]  # chunk offset in the page

            # Append chunk information to chunk_list
            chunk_list.append(
                {
                    "page_number": page_number,
                    "chunk_offset_in_page": chunk_offset_in_page,
                    "chunk_length": len(content),
                    "chunk_offset_in_file": chunk_offset_in_file,
                    "content": chunk_content,
                    "metadata": chunk.metadata,
                }
            )
            chunk_offset_in_file += len(
                content
            )  # increment the chunk offset in the file. This becomes the offset for next chunk.

        # Calculate processing time and file metrics
        e_time = time.time()
        text_size = sum([chunk["chunk_length"] for chunk in chunk_list])
        file_size = os.path.getsize(file_path)
        ratio = file_size / text_size
        seconds = e_time - s_time

        # Return dictionary with file information and processed chunks
        return {
            "file_path": file_path,
            "seconds": seconds,
            "pages": len(documents),
            "chunk_list": chunk_list,
            "file_size": file_size,
            "text_size": text_size,
            "ratio": ratio,
        }

    def process_files(
        self, dir_path, n_threads, file_type=".pdf", chunk_size=1200, chunk_overlap=0
    ):
        """
        Process multiple files in parallel using multiprocessing.

        This method initializes the file processing parameters, retrieves and sorts the files,
        and then processes them in parallel using a multiprocessing pool.

        Args:
            dir_path (str): The directory path containing the files to be processed.
            n_threads (int): The number of threads (processes) to use for parallel processing.
            file_type (str, optional): The file extension to filter for. Defaults to '.pdf'.
            chunk_size (int, optional): The size of each text chunk in characters. Defaults to 1200.
            chunk_overlap (int, optional): The size of the overlap between text chunks. Defaults to 0.

        Returns:
            list: A list of results, where each result is the output of processing a single file.
                The structure of each result depends on the implementation of self._process_file.

        Raises:
            Any exceptions raised by self._init, self._get_files, self._sort_files_by_size,
            or self._process_file methods.

        Note:
            This method assumes the existence of several helper methods in the class:
            _init, _get_files, _sort_files_by_size, and _process_file.
        """
        # Initialize processing parameters
        self._init(dir_path, n_threads, file_type, chunk_size, chunk_overlap)

        # Get list of files to process
        self.files = self._get_files()

        # Sort files by size in decreasing order (for more efficient processing)
        self.files = self._sort_files_by_size(self.files)

        # Process files in parallel using a multiprocessing pool
        with multiprocessing.Pool(self.n_threads) as pool:
            results = pool.map(self._process_file, self.files)

        return results

    def _process_file(self, file):
        """
        Process a single PDF file and generate summary statistics.

        This method reads a PDF file, splits it into chunks, and calculates various
        statistics about the file and its content. It then prints a summary of these
        statistics in a tabular format.

        Args:
            file (str): Path to the PDF file to be processed.

        Returns:
            dict: A dictionary containing processed file information:
                - file_path (str): Path to the processed file
                - pages (int): Number of pages in the PDF
                - chunk_list (list): List of dictionaries, each representing a chunk with metadata
                - file_size (int): Size of the file in bytes
                - text_size (int): Total size of extracted text in characters
                - ratio (float): Ratio of file size to text size
                - seconds (float): Time taken to process the file

        Raises:
            Any exceptions raised by self._read_pdf_file method

        Note:
            This method assumes the existence of self._read_pdf_file method in the class.
            It also uses the tabulate library for formatting output.
        """
        print(f"Processing file {file}...")

        # Process the PDF file
        chunks = self._read_pdf_file(
            file, self.chunk_size, self.chunk_overlap, b_get_content=False
        )

        # Extract relevant information from the chunks dictionary
        file_path = chunks.get("file_path", "")
        pages = chunks.get("pages", 0)
        chunk_list = chunks.get("chunk_list", [])
        file_size = chunks.get("file_size", "0")
        text_size = chunks.get("text_size", "0")
        ratio = chunks.get("ratio", "0")
        seconds = chunks.get("seconds", "0")

        # Prepare data for tabular output
        headers = [
            "Filename",
            "Time (s)",
            "Pages",
            "Chunks",
            "File Size",
            "Text Size",
            "Ratio",
        ]
        file_name = os.path.basename(file_path)
        row = [
            file_name,
            f"{seconds:.2f}",
            str(pages),
            str(len(chunk_list)),
            f"{file_size:,}",
            f"{text_size:,}",
            f"{ratio:.2f}",
        ]

        # Print the summary in tabular format
        print(tabulate([row], headers=headers, tablefmt="grid", showindex=False))
        return chunks


import argparse


def main():
    """
    Main function to process PDF files in multiple folders and generate metadata.

    This function processes all PDF files in specified folders, splits them into chunks,
    and generates metadata for each file. It uses multi-threading for efficient processing
    and saves the results in CSV format.

    Usage:
    python script_name.py <folder_path1> [<folder_path2> ...] <number_of_threads> [--print_metadata]

    Arguments:
    folder_paths: One or more paths to folders containing PDF files
    number_of_threads: Number of threads to use for processing
    --print_metadata: Optional flag. If provided, prints detailed metadata for each file

    Returns:
    None

    Side effects:
    - Prints processing information to console
    - Generates CSV files with results for each processed folder

    Note:
    This function relies on several helper functions and classes:
    _parse_arguments, _folder_path_to_filename, File_Reader, and _print_and_save_results.
    """
    # Parse command line arguments
    folder_paths, number_of_threads, print_metadata = _parse_arguments()
    print(folder_paths, number_of_threads, print_metadata)

    # Process each folder
    for folder_path in folder_paths:
        n_threads = int(number_of_threads)
        result_csv_name = _folder_path_to_filename(folder_path) + ".csv"
        print(f"Processing folder {folder_path} with {n_threads} threads")

        # Initialize file reader and process files
        reader = File_Reader()
        s_time = time.time()
        results = reader.process_files(folder_path, n_threads)
        e_time = time.time()
        total_time = e_time - s_time

        # Print and save results
        _print_and_save_results(results, total_time, result_csv_name, print_metadata)


# Example usage:
if __name__ == "__main__":
    main()
