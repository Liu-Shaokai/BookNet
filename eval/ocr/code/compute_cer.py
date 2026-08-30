import argparse
import os

import editdistance
import numpy as np

IMG_LIST = ["03", "05", "07", "08", "10", "17", "18", "22", "26", "28", "29", "40", "42", "43", "47",
            "48", "50", "51", "52", "55", "57", "65", "67", "82", "83", "87", "89", "91", "96", "100"]


def cal_cer_ed(path_scan, path_method):
    ed_list = []
    cer_list = []
    for img_id in IMG_LIST:
        with open(os.path.join(path_scan, img_id + '.txt'), encoding='utf-8') as f:
            gt = f.read()
        with open(os.path.join(path_method, img_id + '.txt'), encoding='utf-8') as f:
            pred = f.read()
        ed = editdistance.eval(pred, gt)
        ed_list.append(ed)
        cer_list.append(ed / len(gt))
    return np.mean(ed_list), np.mean(cer_list)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", required=True)
    ap.add_argument("--method-dir", required=True)
    args = ap.parse_args()

    ED, CER = cal_cer_ed(args.scan_dir, args.method_dir)
    print(f"ED={ED:.4f}  CER={CER:.4f}")
