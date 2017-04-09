import csv
import datafiled as df
from sklearn.model_selection import train_test_split

def getTrainDate():
    samples = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (line[0] != 'center'):
                samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=df.TrainTestSplitSize)

    return train_samples, validation_samples
