# I_fetch_pubmed.py --> ran on modal

import os
import pickle
import shutil
from tqdm import tqdm
from ftplib import FTP
from time import sleep
import pubmed_parser as pp
from urllib import request
from random import shuffle
from itertools import chain
from multiprocessing import Pool
from collections import defaultdict


num_workers = 10
base_url = 'https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'

medline_folder = 'pmid2contents'
os.makedirs(medline_folder, exist_ok=True)


def clean_title(title):
    """
    :param title:
    :return: Basic text cleaning for title
    """
    title = ' '.join(title) if isinstance(title, list) else title

    if title.startswith('['):title = title[1:]
    if title.endswith(']'): title = title[:-1]
    if title.endswith('.'): title = title[:-1]
    if title.endswith(']'): title = title[:-1]
    return title.lower() + ' .'


def clean_abstract(abstract):
    """
    :param abstract:
    :return: Basic text cleaning for abstract
    """
    if abstract.endswith('.'): abstract = abstract[:-1] + ' .'
    return abstract.lower()


def get_medline_files_path():
    """
    :return: helper function to get medline file names
    """
    file_names = []
    with FTP('ftp.ncbi.nlm.nih.gov') as ftp:
        ftp.login()
        lines = []
        ftp.dir('pubmed/baseline', lines.append)
        for i in lines:
            tokens = i.split()
            name = tokens[-1]
            if name.endswith('.gz'):
                file_names.append(name)
    return file_names


def medline_download(renew=False):
    print('Downloading Medline XML files ...')
    file_names = get_medline_files_path()[:30] # Remove this for full-scale operation
    for f_name in tqdm(file_names):
        if not os.path.isfile(os.path.join(medline_folder, f_name)) or renew:
            if f_name not in os.listdir(medline_folder):
                with request.urlopen(os.path.join(base_url, f_name)) as response, open(os.path.join(medline_folder, f_name), 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                    sleep(1)


def medline_parser(med_xml):
    dicts_out = pp.parse_medline_xml(os.path.join(medline_folder, med_xml),
                                     year_info_only=False,
                                     nlm_category=False,
                                     author_list=False,
                                     reference_list=False)

    pack = []
    for i in dicts_out:
        pmid = i['pmid']
        c_title = clean_title(i['title'])
        title = c_title if len(c_title)>10 else None # ignore noise titles

        c_abstract = clean_abstract(i['abstract'])
        abstract = c_abstract if len(c_abstract)>10 else None # ignore noise abstract

        if len(i['mesh_terms']):
            mesh_terms = [x.strip().split(':')[1].lower() for x in i['mesh_terms'].split(';')]
        else:
            mesh_terms = None

        if all([title, abstract, mesh_terms]):
            pack.append((pmid, title, abstract, mesh_terms))
    return pack


def multi_process_medline():
    """
    :return: list of pickle files in which pmids are mapped to their mesh terms, titles and abstracts (strings)
    """
    print('Processing XML files ...')
    xml_files = [xml_file for xml_file in os.listdir(medline_folder) if xml_file.endswith('.xml.gz')]
    shuffle(xml_files) #load-balance files with different sizes
    for idx in tqdm(range(0, len(xml_files), 10)):
        xml_files_batch = xml_files[idx: idx + 10]
        with Pool(processes=num_workers) as pool:
            pmid2content_map_all = pool.map(medline_parser, xml_files_batch)
        pmid2content_map_all = list(chain(*pmid2content_map_all))

        pmid2content = defaultdict(set)
        for entry in pmid2content_map_all:
            pmid2content[entry[0]] = entry[1:]

        with open(os.path.join(medline_folder, 'pmid2content%d.pkl' % idx), 'wb') as f:
            pickle.dump(pmid2content, f)
        pmid2content.clear()
    for gz_file in os.listdir(medline_folder): # remove processed files
        if gz_file.endswith('.gz'):
            os.remove(os.path.join(medline_folder, gz_file))



if __name__ == "__main__":
    medline_download()
    multi_process_medline()