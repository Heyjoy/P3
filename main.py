import utils
import models

train_samples, validation_samples = utils.getTrainDate();
train_generator = utils.generator(train_samples, batch_size=32)
validation_generator = utils.generator(validation_samples, batch_size=32)
#print(train_samples[0])
model = models.end2endNiv()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch= len(train_samples),
                                     validation_data=validation_generator,
                                     validation_steps= len(validation_samples),
                                     epochs=3,
                                     verbose =1)

model.save('model.h5')
plotHistroy(history_object)
