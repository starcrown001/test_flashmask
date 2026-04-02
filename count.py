from collections import Counter

consumer_counts = Counter()
producer_counts = Counter()

with open('debug.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('consumer blockid:'):
            blockid = line.split(':')[1].strip()
            consumer_counts[blockid] += 1
        elif line.startswith('producer blockid:'):
            blockid = line.split(':')[1].strip()
            producer_counts[blockid] += 1

all_blockids = set(consumer_counts.keys()) | set(producer_counts.keys())

print("Blockid 个数不匹配的 id 及其 consumer/producer 个数：")
for blockid in sorted(all_blockids, key=int):
    consumer_num = consumer_counts.get(blockid, 0)
    producer_num = producer_counts.get(blockid, 0)
    if consumer_num != producer_num:
        print(f'blockid: {blockid}, consumer: {consumer_num}, producer: {producer_num}')