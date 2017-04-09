import csv
import datafiled

def getTrainDate():
    samples = []
    with open('../data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if (line[0] != 'center'):
                samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=TrainTestSplitSize)

    return train_samples, validation_samples
