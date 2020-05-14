import os
from electra_model import trainer

with open('../labels.json') as f:
    labels = json.load(f)

intent_class_num = len(labels['intent'])
entity_class_num = len(labels['entity']) + 1  ##consider XO type


trainer.train(
    file_path='../nlu_goldenset.md',

    #training args
    train_ratio=0.8,
    batch_size=128,
    optimizer="Adam",
    intent_optimizer_lr=1e-5,
    entity_optimizer_lr=2e-5,
    checkpoint_path=os.getcwd(),
    max_epochs=6,

)