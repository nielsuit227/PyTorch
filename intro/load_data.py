from torchtext.datasets import IMDB


train_iter = IMDB(split='train')
print(type(train_iter))
for seq, label in train_iter:
    print(seq, label)