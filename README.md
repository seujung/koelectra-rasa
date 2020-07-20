# KoELECTRA based DIET Model
Dual Intent Entity Transformer Pytorch version based on KoELECTRA

It is implemented [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) based module


## Model Architecture



## Dataset
- nlu.md
```
## intent:check_balance
- what is my balance <!-- no entity -->
- how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
- Could I pay in [yen](currency)? 

## intent:greet
- hey
- hello

```

- labels.json

```
{
    "intent" : ['check_balance', 'greet],
    "entity" : ['sourec_account', currency]
}

```


##


## How to train
---
1. Training 

    ```
    import json
    from DIET import trainer

    with open('labels.json') as f:
    labels = json.load(f)

    intent_class_num = len(labels['intent'])
    entity_class_num = len(labels['entity']) * 2 + 1  ## consider BIO type

    trainer.train(
        file_path='./nlu.md'
        #training args
        train_ratio=0.8,
        batch_size=256,
        intent_class_num = intent_class_num,
        entity_class_num = entity_class_num,
        optimizer="AdamW",
        intent_optimizer_lr=3e-5,
        entity_optimizer_lr=4e-5,
        checkpoint_path='electra_diet_log',
        max_epochs=20,
        gpu_num=1,
        lower_text=True,
        early_stop=True,
        report_nm ="electra_report.json"
        **kwargs
    )
    ```

    file_path indicate markdown format NLU dataset which follow below [RASA NLU training data format](https://rasa.com/docs/rasa/nlu/training-data-format/#markdown-format)

    All parameters in trainer including kwargs saved as a model hparams

    User can check these paramters via checkpoint tensorboard logs(use lightning_logs folder to user tensorboard)
    ![tensorboard log](img/tensorboard_log.PNG)

2. Inference

    ```
    from DIET import Inferencer

    inferencer = Inferencer(checkpoint_path)
    inferencer.inference(text: str, intent_topk=5)
    ```

    As this repository model is implemented based on pytorch-lightning, it generate checkpoint file automatically(user can set checkpoint path in training step)

    After setting checkpoint path, query text to inferencer. Result contain intent_rank, user can set n-th rank confidences of intents.

    Inference result will be like below
    ```
    {
        "text": "오늘 서울 날씨 어때?",
        "intent": {
            "confidence": 0.6323,
            "name": "ask_weather"
        },
        "intent_ranking": [
            {
                "confidence": 0.6323,
                "name": "ask_weather"
            },
            ...
        ],
        "entities": [
            {
                "start": 3,
                "end": 4,
                "value": "서울",
                "entity": "location"
            },
            ...
        ]
    }
    ```

## How it works
---

The model in this repository refered from Rasa DIET classifier.

[this blog ](https://ryanong.co.uk/2020/04/10/day-101-in-depth-study-of-rasas-diet-architecture/) explain how it works in Rasa framework.

But more simple implementation & fast training, inference,
There are several changes int here.

1. There is no CRF layer ahead TransformerEncoder layer

    In real training situation, CRF training pipeline takes a lot of training time. But it can not sure CRF model really learn token relation well or it really need(I guess transformer self-attention do similar things)

2. It takes character tokenzier for enhancing korean language parsing. 

    Differ from English or other languages. Korean's character can be joined or splitted in character themselves. Considering this feature, I applied character based tokenizer

3. There is no mask loss.

    Relating upper difference, it doesn't use any pre-trained embedding and tokenizer. So masking techinique is hard to apply.



