import os
import time
from termcolor import colored


def split_markdown(input_file):
    with open(input_file, "r") as file:
        content = file.readlines()

    sections = []
    current_section = []

    for line in content:
        if line.startswith("## "):
            if current_section:
                sections.append("".join(current_section).strip())
            current_section = [line]
        else:
            current_section.append(line)

    if current_section:
        sections.append("".join(current_section).strip())

    return sections


def write_to_temp_file(content, index):
    # Skip creating the file if the content is empty
    if not content.strip():
        return None

    temp_file_name = f"temp_section_{index}.md"
    # Add "\pagenumbering{gobble}" to the beginning of the file to remove page numbering in the PDF after conversion
    content_with_pagenumbering = "\\pagenumbering{gobble}\n" + content
    with open(temp_file_name, "w") as temp_file:
        temp_file.write(content_with_pagenumbering)
    return temp_file_name


def convert_to_pdf(input_file):
    output_file = input_file.replace(".md", ".pdf")
    print(f"Converting {input_file} to {output_file}")
    os.system(
        f'pandoc {input_file} -o {output_file} --pdf-engine=xelatex -V mainfont="DejaVu Sans"'
    )

    return output_file


def combine_pdfs(pdf_files, output_file):
    pdf_files_joined = " ".join(pdf_files)
    # use pdfunite
    command = f"pdfunite {pdf_files_joined} {output_file}"
    print(f"\npdfunite command:\n{command}")
    os.system(command)
    # command = f"pdftk {pdf_files_joined} cat output {output_file}"
    # subprocess.run(['pdftk', pdf_files_joined, 'cat', 'output', output_file])


def delete_temp_files(temp_files):
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)


def main(input_file):
    sections = split_markdown(input_file)
    temp_files = []
    pdf_files = []

    for index, section in enumerate(sections):
        temp_file = write_to_temp_file(section, index)
        if temp_file:
            temp_files.append(temp_file)

    for temp_file in temp_files:
        pdf_file = convert_to_pdf(temp_file)
        if pdf_file:  # Only append non-empty PDF files to the list
            pdf_files.append(pdf_file)

    output_pdf = input_file.replace(".md", ".pdf")
    combine_pdfs(pdf_files, output_pdf)

    # wait 5 seconds
    time.sleep(5)
    # delete temp markdown files
    delete_temp_files(temp_files)
    # delete pdf files
    delete_temp_files(pdf_files)
    # delete missfont.log if the file exists
    if os.path.exists("missfont.log"):
        os.remove("missfont.log")


if __name__ == "__main__":
    this_root = os.path.dirname(os.path.realpath(__file__))
    
    print(
        colored("\n[NOTE]", "light_red"),
        f"Ensure that the markdown source file is in the same directory as this script ({this_root})",
    )
    print(
        colored("[NOTE]", "light_red"),
        "Also make sure all pictures embedded in that markdown are also in that directory, otherwise they won't be included in the PDF.",
    )
    print(
        colored("[NOTE]", "light_red"),
        "Make sure to put a blank line before any numbered/ordered lists in the markdown file\n",
    )
    
    all_files = os.listdir(this_root)
    md_count = 0
    for file in all_files:
        if file.endswith(".md"):
            md_count += 1
    if md_count == 1:
        file = [file for file in all_files if file.endswith(".md")][0]
        print(
            colored("\n[INFO]", "cyan"),
            f"Found {md_count} markdown file in this script's directory:",
            file
        )
    else:
        file = input(
            colored(
                "\nEnter the filename of the md file in this script's dir:\n> ",
                "light_green",
            )
        )
    
    if not file.endswith(".md"):
        print(colored("\n[ERROR]", "red", attrs=["bold"]), "File must be a markdown file. Automatically appending '.md' to the filename. Disable if trying to use script with other type of input file.")
        file += ".md"

    pdf_file_name = file.replace(".md", ".pdf")
    if os.path.exists(os.path.join(this_root, pdf_file_name)):
        os.remove(os.path.join(this_root, pdf_file_name))
        print(
            colored("\n[INFO]", "cyan"),
            f"Removed existing {pdf_file_name} in this script's directory.",
        )
    path = os.path.join(this_root, file)
    main(path)
