import os
import pickle
import shutil
from tqdm import tqdm
from ftplib import FTP
from time import sleep
import pubmed_parser as pp
from urllib import request
from collections import defaultdict

base_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
medline_folder = 'pmid2contents'
os.makedirs(medline_folder, exist_ok=True)


def clean_title(title):
    title = ' '.join(title) if isinstance(title, list) else title
    if title.startswith('['):
        title = title[1:]
    if title.endswith(']'):
        title = title[:-1]
    if title.endswith('.'):
        title = title[:-1]
    return title.lower() + ' .'


def clean_abstract(abstract):
    if abstract.endswith('.'):
        abstract = abstract[:-1] + ' .'
    return abstract.lower()

# modified version of the script (with AI), because for me it was not possible to run "I_fetch_pubmed.py"
def get_medline_files_path(limit=20):
    file_names = []
    with FTP('ftp.ncbi.nlm.nih.gov') as ftp:
        ftp.login()
        lines = []
        ftp.dir('pubmed/baseline', lines.append)
        for line in lines:
            name = line.split()[-1]
            if name.endswith('.gz'):
                file_names.append(name)
    return file_names[:limit]  


def medline_download(renew=False):
    print('Downloading Medline XML files ...')
    file_names = get_medline_files_path()
    for f_name in tqdm(file_names):
        local_path = os.path.join(medline_folder, f_name)
        if not os.path.isfile(local_path) or renew:
            with request.urlopen(os.path.join(base_url, f_name)) as response, open(local_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            sleep(1)  


def medline_parser_stream(med_xml):
    pmid2content = {}
    file_path = os.path.join(medline_folder, med_xml)

    # parse_medline_xml returns a generator if we read in chunks (less memory)
    dicts_out = pp.parse_medline_xml(file_path,
                                     year_info_only=False,
                                     nlm_category=False,
                                     author_list=False,
                                     reference_list=False)

    for entry in dicts_out:
        pmid = entry['pmid']
        title = clean_title(entry['title'])
        abstract = clean_abstract(entry['abstract'])
        if len(title) < 10 or len(abstract) < 10 or not entry['mesh_terms']:
            continue
        mesh_terms = [x.strip().split(':')[1].lower() for x in entry['mesh_terms'].split(';')]
        pmid2content[pmid] = {
            'title': title,
            'abstract': abstract,
            'mesh_terms': mesh_terms
        }

    # Delete XML file after processing to save space
    os.remove(file_path)
    return pmid2content


def process_medline_files():
    print('Processing XML files ...')
    all_files = sorted([f for f in os.listdir(medline_folder) if f.endswith('.xml.gz')],
                       key=lambda x: os.path.getsize(os.path.join(medline_folder, x)))
    all_data = {}

    # Process one file at a time to save RAM
    for xml_file in tqdm(all_files[:2]): 
        batch_data = medline_parser_stream(xml_file)
        all_data.update(batch_data)

    # Save final pickle
    out_path = os.path.join(medline_folder, 'pmid2content.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_data, f)
    print(f"Saved {len(all_data)} PMIDs.")


if __name__ == "__main__":
    medline_download()
    process_medline_files()
