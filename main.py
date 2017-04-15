import utils
import models
import datafield as df
# get/split the train and validation samples from the log file
train_samples, validation_samples = utils.getTrainDate();

# setup two kind generator
train_generator = utils.generator(train_samples, batch_size=df.BatchSize)
validation_generator = utils.generator(validation_samples, batch_size=df.BatchSize)

# use nivida end to end concept
model = models.end2endNiv()
model.compile(loss='mse', optimizer='adam')

# start training the model and record
history_object = model.fit_generator(train_generator,
                                         samples_per_epoch= len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples= len(validation_samples),
                                         nb_epoch=df.N_EPOCH,
                                         verbose =df.Verbose)
model.save('model.h5')

#print out the training reslut.
utils.plotHistroy(history_object)
