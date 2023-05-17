import argparse
import json
from datetime import date
from os import listdir, mkdir
from os.path import isdir, isfile, join, splitext
from xml.dom.minidom import parse, parseString


def get_article_txt(txtpath, fname):
    fname_txt = fname.rstrip("knowtator.xml") + ".txt"
    file = join(txtpath, fname_txt)
    with open(file, "r") as f:
        return f.read()


def xml2json(dirpath, txtpath, outdir, prefix, agg):
    '''converts original knowtator xml annotation pairs (mention, GO_ID)
    to either a set of json files (default) or an aggregate of all annotations
    '''
    annotated_pairs = []

    # iterate over all *.xml files in dirpath
    for fname in listdir(dirpath):
        fpath = join(dirpath, fname)
        if (splitext(fname)[1] == ".xml"): 
            aid2mention = {}
            if not agg: annotated_pairs = []
            content = parse(fpath)

            # get article txt
            article_txt = get_article_txt(txtpath, fname)

            # create annotation_id to mention map
            annotations = content.getElementsByTagName("annotation")
            for annot in annotations:
                # annotation_id
                annot_id = annot.getElementsByTagName("mention")[0]
                annot_id = annot_id.getAttribute("id")
                # get span
                span_elem = annot.getElementsByTagName("span")[0]
                start = int(span_elem.getAttribute("start"))
                end = int(span_elem.getAttribute("end"))
                span = (start, end)
                # get mention txt
                mention = annot.getElementsByTagName("spannedText")[0]
                mention = mention.childNodes[0].nodeValue
                aid2mention[annot_id] = (mention, span)
                
            # match mention to GO ID (annotated pairs)
            mention_classes = content.getElementsByTagName("classMention")
            for mclass in mention_classes:
                annot_id = mclass.getAttribute("id").rstrip()
                onto_id = mclass.childNodes[1].getAttribute("id")
                onto_id = int(onto_id.lstrip(f"{prefix}:").rstrip())
                mention, span = aid2mention[annot_id]
                annotated_pairs.append([mention, span, onto_id])

            # save processed json
            if not agg:
                opath = join(outdir, fname.split('.')[0]+'.json')
                meta = {"date": str(date.today()), "size": len(annotated_pairs)}
                with open(opath, 'w') as f: 
                    data = {
                        "data": annotated_pairs, "article_txt": article_txt,
                        "meta": meta
                    }
                    json.dump(data, f)

    # save aggregate
    if agg:
        opath = join(outdir, "aggregated.json")
        meta = {"date": str(date.today()), "size": len(annotated_pairs)}
        with open(opath, 'w') as f: 
            json.dump({"data": annotated_pairs, "meta": meta}, f)
         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirpath",
        default="../../datasets/CRAFT/CRAFT_full/CL/CL/knowtator/",
        help="path to CRAFT knowtator directory"
    )
    parser.add_argument(
        "--txtpath",
        default="../../datasets/CRAFT/articles/txt/",
        help="path to CRAFT full txt articles directory"
    )
    parser.add_argument(
        "--outdir",
        default="../../datasets/CRAFT/CL/json/",
        help="path to output directory"
    )
    parser.add_argument(
        "--aggregate",
        default=True,
        help="whether or not to aggregate into one .json file"
    )
    parser.add_argument(
        "--ontoprefix",
        default="CL"
    )
    args = parser.parse_args()
    assert(isdir(args.dirpath))
    if not isdir(args.outdir): mkdir(args.outdir)

    xml2json(
        args.dirpath, 
        args.txtpath,
        args.outdir,
        args.ontoprefix,
        args.aggregate
    )

