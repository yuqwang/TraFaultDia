import re

def clean_trainticket(sentence):
    cleaned_sentence = sentence.lower()
    cleaned_sentence = re.sub(r'\b[0-9a-f]{8}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{12}\b', 'routeid',
                              cleaned_sentence)
    cleaned_sentence = re.sub(r'gaotie\w*', 'traintype', cleaned_sentence)
    cleaned_sentence = re.sub(r'dongche\w*', 'traintype', cleaned_sentence)
    cleaned_sentence = re.sub(r'(kuaisu)', 'traintype', cleaned_sentence)
    cleaned_sentence = re.sub(r'(tekuai)', 'traintype', cleaned_sentence)
    cleaned_sentence = re.sub(r'(shang hai|shanghai)', 'cityname', cleaned_sentence)
    cleaned_sentence = re.sub(r'(nan jing|nanjing)', 'cityname', cleaned_sentence)
    cleaned_sentence = re.sub(r'(su zhou|suzhou)', 'cityname', cleaned_sentence)
    cleaned_sentence = re.sub(r'(tai yuan|taiyuan)', 'cityname', cleaned_sentence)
    cleaned_sentence = re.sub(r'(zhen jiang|zhenjiang)', 'cityname', cleaned_sentence)
    cleaned_sentence = re.sub(r'(wu xi|wuxi)', 'cityname', cleaned_sentence)
    cleaned_sentence = re.sub(r'(spicy hot noodles)', 'food', cleaned_sentence)
    cleaned_sentence = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_sentence)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
    return cleaned_sentence

def clean_log_message_onlineboutique(sentence):
    # Clean "PlaceOrder user_id="a5df2f22-43ac-4006-a284-77f6beb30eac" user_currency="JPY" successfully" format
    #cleaned_sentence = re.sub(r'(\w+?)="[^"]*?"', r'\1', sentence)
    #cleaned_sentence = re.sub(r'(\w+?)=.*?(\s|$)', r'\1\2', sentence)
    cleaned_sentence = sentence.lower()
    cleaned_sentence = re.sub(r"list recommendations product_ids.*?$", "list recommendations product_ids]", cleaned_sentence)
    # Clean "get results: {...}" format
    cleaned_sentence = re.sub(r"get results:.*?$", "get results", cleaned_sentence)
    cleaned_sentence = re.sub(r'\b[a-z0-9]{10}\b', 'product id', cleaned_sentence)
    cleaned_sentence = re.sub(r'\b[0-9a-f]{8}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{4}\-[0-9a-f]{12}\b', '', cleaned_sentence)
    cleaned_sentence = re.sub(r'context_words=\[.*?\]', 'context_words=producttype', cleaned_sentence)
    cleaned_sentence = re.sub(r'".*?"', '', cleaned_sentence)
    cleaned_sentence = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b', '', cleaned_sentence)
    cleaned_sentence = re.sub(r"jpy", "currency code", cleaned_sentence)
    cleaned_sentence = re.sub(r"usd", "currency code", cleaned_sentence)
    cleaned_sentence = re.sub(r"eur", "currency code", cleaned_sentence)
    cleaned_sentence = re.sub(r"cad", "currency code", cleaned_sentence)
    cleaned_sentence = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_sentence)
    cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence)
    return cleaned_sentence.strip()

def clean_span_url_onlineboutique(sentence):
    cleaned_sentence = sentence.lower()
    cleaned_sentence = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_sentence)
    return cleaned_sentence.strip()