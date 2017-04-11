import utils
import models


train_samples, validation_samples = utils.getTrainDate();
train_generator = utils.generator(train_samples, batch_size=32)
validation_generator = utils.generator(validation_samples, batch_size=32)
#print(train_samples[0])
model = models.end2endNiv()
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                                     samples_per_epoch= len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples= len(validation_samples),
                                     nb_epoch=3,
                                     verbose =1)
model.save('model.h5')
utils.plotHistroy(history_object)
