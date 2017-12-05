import os

def main():
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    # train = os.path.join(PROJECT_ROOT, "data/kddcup.data")
    # test = os.path.join(PROJECT_ROOT, "data/corrected")
    train_10 = os.path.join(PROJECT_ROOT, "data/kddcup.data_10_percent")
    # out = os.path.join(PROJECT_ROOT, "data/train.data")
    # out = os.path.join(PROJECT_ROOT, "data/test.data")
    out = os.path.join(PROJECT_ROOT, "data/train10.data")


    try:
        kdd_read = open(train_10, "r")
    except:
        print('Can not open the file')

    label = load_label()

    kdd_readlines = kdd_read.readlines()
    for k in kdd_readlines:
        k = k.split(",")[-1].split(".\n")[0]
        print(k)
        label[k] = label[k] + 1
    kdd_read.close()

    for (k, v) in label.items():
        print(k, "\t", v)

def load_label():
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    label = os.path.join(PROJECT_ROOT, "data/label")
    label_dict = {}
    try:
        label_read = open(label, "r")
    except:
        print('Can not open the file')

    label_lines = label_read.readlines()
    for k in label_lines:
        key, value = k.split("\t")
        label_dict[key] = 0
    label_read.close()
    return label_dict

if __name__ == "__main__":
    main()