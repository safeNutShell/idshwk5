import math
from sklearn.ensemble import RandomForestClassifier

domainList = []
domainPredict = []


class Domain:
    def __init__(self, _name, _label, _length, _num, _entropy):
        self.name = _name
        self.label = _label
        self.length = _length
        self.num = _num
        self.entropy = _entropy

    def returnData(self):
        return [self.length, self.num, self.entropy]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

    def returnName(self):
        return self.name


def numCount(domain):
    num = 0
    for i in domain:
        if i.isdigit():
            num += 1
    return num


def calEntropy(text):
    h = 0.0
    count = 0
    letter = [0] * 26
    text = text.lower()
    for i in range(len(text)):
        if text[i].isalpha():
            letter[ord(text[i]) - ord('a')] += 1
            count += 1
    for i in range(26):
        p = 1.0 * letter[i] / count
        if p > 0:
            h += -(p * math.log(p, 2))
    return h


def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(name)
            num = numCount(name)
            entropy = calEntropy(name)
            domainList.append(Domain(name, label, length, num, entropy))


def loadData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            length = len(line)
            num = numCount(line)
            entropy = calEntropy(line)
            domainPredict.append(Domain(line, "dga", length, num, entropy))


def main():
    initData("train.txt")
    loadData("test.txt")
    featureMatrix = []
    labelList = []
    for item in domainList:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())
    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    f = open("result.txt", "w")
    for item in domainPredict:
        s = item.returnName() + ","
        if clf.predict([item.returnData()]) == 0:
            s = s + "notdga"
        else:
            s = s + "dga"
        s = s + '\n'
        f.write(s)
    f.close()


if __name__ == '__main__':
    main()
