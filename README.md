# 6-pic-vote-mobilenet

## Config

**USE_CUDA** : Whether to use GPU.
**LOAD_SAVED_MOD** : Whether to load saved model.
**SAVE_TEMP_MODEL** : Whether to save temporary model while training.
**SAVE_BEST_MODEL** : Whether to save best model while training.
**BEST_MODEL_BY_LOSS** : Evaluate whether a model is the optimal one by loss or accuracy.
**PRINT_BAD_CASE** : Whether to print the bad case while predicting.
**RUNNING_ON_JUPYTER** : Whether the program is running on a Jupyter Notebook.
**START_VOTE_PREDICT** : Whether to start vote predicting or training.
**START_PREDICT** : Whether to start predicting or training.
**TRAIN_ALL** : Whether to train in all of the data (train_set and val_set).
**TEST_ALL** : Whether to validate all of the data (train_set and val_set).
**TO_MULTI** : Whether to use multiple GPU, if available.
**ADD_SUMMARY** : Whether to add net graph into tensorboard summary.
**SAVE_PER_EPOCH** : Save your temp model every n epoch.
**BATCH_SIZE** : Batch size of training.
**VAL_BATCH_SIZE** : Batch size of validating.
**TENSOR_SHAPE** : Tensor shape of your input (batch dim is not included).
**DATALOADER_TYPE** : Dataloader type of your data (only `ImageFolder`, `SamplePairing`, `SixBatch`)
**OPTIMIZER** : Optimizer type. It is a string which is not case sensitive.Currently `Adam` and `SGD` are supported. Add new optimizer in the ./models/BasicModule.py -> get_optimizer()
**SGD_MOMENTUM** : The momentum if SDG is chosen as optimizer.
**TRAIN_DATA_RATIO** : The Train_Val data ratio.
**NUM_EPOCHS** : The epochs you want to train your model.
**NUM_CLASSES** : The number of your input data's class.
**NUM_VAL** : The number of your validation data.
**NUM_TRAIN** : The number of your train data.
**TOP_NUM** : If top n accuracy is ok for your result, put the `n` here.
**NUM_WORKERS** : Number of workers used in the DataLoader.
**CRITERION** : The Loss Class used in your training process, which is an instance of a Loss Class.
**LEARNING_RATE** : Learning rate used in your optimizer.
**TOP_VOTER** : Top `n` votes in the 6 picture generated will count for the final result.
**NET_SAVE_PATH** : Where to save your trained model.
**TRAIN_PATH** : Where your training set is located.
**VAL_PATH** : Where your validating set is located.
**CLASSES_PATH** : Where to save your classes' name.
**MODEL_NAME** : The name of your model.
**PROCESS_ID** : The `ID` of the current training process, which is the marker of the trained models. Please change it when some config or crucial code is altered!
**SUMMARY_PATH** : Where to save your tensorboard summary.