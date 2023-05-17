'''Scrapes pubmed abstracts from NIH to populate signature full description'''
import argparse
import json
import requests
import threading
import logging
import queue
from os.path import isfile
from requests.exceptions import ConnectionError
BASE_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json/"


def pull_text_worker(processed_dict: dict, q: queue.Queue, v: bool):
    '''queries ncbi's pubmed api with a pmid 
    and extracts all meaningful text
    '''
    while True:
        # break if queue is empty
        try: sigID, pmid, key = q.get(block=False)
        except queue.Empty: break 
        if v: print(sigID, pmid)

        # retrieve from API
        req = BASE_URL + pmid + "/ascii"
        try: res = requests.get(req, timeout=5)
        except ConnectionError: break

        # retrieve texts from result
        texts = []
        if res.status_code == 200:
            texts = [p["text"] for p in res.json()["documents"][0]["passages"]]
        texts = ''.join(texts)

        # add to dict (thread-safe)
        processed_dict["signatures"][sigID] = pmid
        if pmid not in processed_dict["texts"].keys():
            processed_dict["texts"][pmid] = {
                "title": key,
                "text": texts,
                "retrieved_sucess": True if res.status_code==200 else False 
            }
        q.task_done()


def main(sigpath: str, outpath: str, thread_num: int, verbose: bool):
    '''processes signature collection and retrieves text 
    data from NCBI's API
    '''
    with open(sigpath, "r") as f:
        sig_dict = json.load(f)

    q = queue.Queue()
    processed_dict = {
        "signatures": {},
        "texts": {}
    }
    for key, val in sig_dict.items():
        try:
            sigID = val["systematicName"]
            pmid = val["pmid"]
        except: continue
        # add to queue
        q.put((sigID, pmid, key))

    # start threadpool
    threads = []
    for _ in range(thread_num):
        threads.append(threading.Thread(
            target = pull_text_worker, 
            args = (processed_dict, q, verbose))
        )
    for t in threads: t.start()
    for t in threads: t.join()

    # save
    with open(outpath, "w") as f:
        json.dump(processed_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument(
        "--sigpath",
        default = "../../signatures/c7.vax.v2022.1.Hs.json",
        help = "path to raw signatures json from an MsigDB collection"
    )
    parser.add_argument(
        "--outpath",
        default = "../../signatures/c7_2022.json",
        help = "filepath to processed signatures json"
    )
    parser.add_argument(
        "--thread_num",
        default = 4,
        help = "number of threads"
    )
    parser.add_argument(
        "--verbose",
        default = False,
        help = "print logging info"
    )
    args = parser.parse_args()
    main(args.sigpath, args.outpath, args.thread_num, args.verbose)
