from scipy.stats import multivariate_normal as m
from sklearn.preprocessing import normalize
import numpy as np

# Load the data
import scipy.io
mat = scipy.io.loadmat('data/digit-dataset/train.mat')
images = np.reshape(mat['train_image'], (1, -1, 60000))[0].T
labels = mat['train_label']

# Shuffle the data in parallel
p = np.random.permutation(len(images))
images, labels = images[p], labels[p]

def ccc(images, labels, average_cov):
    # Model each class as a Gaussian
    classes = list(set(labels[:,0]))
    means, covs, priors = [], [], []
    ccds = []
    for c in classes:
        # Filter out the class, then normalize
        locs = np.array([label[0] == c for label in labels])
        data = normalize(images[locs].astype(float), norm='l2')
        
        # print("one")

        mean = sum(data) / len(data)
        cov = np.cov(data, rowvar=0)
        
        ccds.append(m.logpdf(test, mean, cov + 0.001 * np.eye(len(cov))))

    
    a = [max((ccds[d][i], d) for d in range(10))[1] for i in range(5000)]
    return sum(x == y for x, y in zip(a, answers))

# Load the test data
mat = scipy.io.loadmat('data/digit-dataset/test.mat')
test = np.reshape(mat['test_image'], (1, -1, 5000))[0].T
answers = mat['test_label']

print(ccc(images, labels, False))

# %time print(score(images, labels, test, answers, False))