from pyhanlp import HanLP

cases = []
document = ''
with open("./data/1.txt", "r") as f:
    for line in f:
        cases.append(line)
    f.seek(0)
    document = f.read()

# segmentation
# print(cases)
for item in cases:
    print(HanLP.segment(item))

# keyword
# print(document)
print(HanLP.extractKeyword(document, 2))

# summary
print(HanLP.extractSummary(document, 3))
