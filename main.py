import utils
import models
import datafiled as df

train_samples, validation_samples = utils.getTrainDate();
#train_samples = samplesProcess(train_samples_raw)

train_generator = utils.generator(train_samples, batch_size=df.BatchSize)
validation_generator = utils.generator(validation_samples, batch_size=df.BatchSize)
#print(train_samples[0])
model = models.end2endNiv()
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                                         samples_per_epoch= 6*len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples= 6*len(validation_samples),
                                         nb_epoch=df.N_EPOCH,
                                         verbose =df.Verbose)
model.save('model.h5')
utils.plotHistroy(history_object)
