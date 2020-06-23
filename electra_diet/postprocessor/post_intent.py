def post_process(intent, entities):

    ##로밍 관련 후처리 로직
    if intent['name'] == 'intent_로밍_요금제_가입_멀티턴':
        existed_entity_list = ['RoamingPlan_Hi', 'RoamingPlan']
        
        cnt = 0
        for e in entities:
            if e['entity'] in existed_entity_list:
                cnt +=1
        
        if cnt == 0:
            intent['name'] = 'intent_로밍_요금제_추천_문의_멀티턴'
            intent['confidence'] = 0.99
    
    return intent, entities
        
