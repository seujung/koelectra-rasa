import os
from electra_model import trainer

trainer.train(
    file_path='../nlu.md',

    #training args
    train_ratio=0.8,
    batch_size=128,
    optimizer="Adam",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=6,

)