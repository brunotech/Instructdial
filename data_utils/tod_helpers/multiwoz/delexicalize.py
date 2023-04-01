import re

import simplejson as json

from .nlp import normalize

digitpat = re.compile('\d+')
timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat2 = re.compile("\d{1,3}[.]\d{1,2}")

# FORMAT
# domain_value
# restaurant_postcode
# restaurant_address
# taxi_car8
# taxi_number
# train_id etc..


def prepareSlotValuesIndependent():
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']
    requestables = ['phone', 'address', 'postcode', 'reference', 'id']
    dic = []
    dic_area = []
    dic_food = []
    dic_price = []

    # read databases
    for domain in domains:
        try:
            with open(f'data/multi-woz/db/{domain}_db.json', 'r') as fin:
                db_json = json.load(fin)
            for ent in db_json:
                for key, val in ent.items():
                    if val in ['?', 'free']:
                        continue
                    if key == 'address':
                        dic.append((normalize(val), f'[{domain}_address]'))
                        if "road" in val:
                            val = val.replace("road", "rd")
                            dic.append((normalize(val), f'[{domain}_address]'))
                        elif "rd" in val:
                            val = val.replace("rd", "road")
                            dic.append((normalize(val), f'[{domain}_address]'))
                        elif "st" in val:
                            val = val.replace("st", "street")
                            dic.append((normalize(val), f'[{domain}_address]'))
                        elif "street" in val:
                            val = val.replace("street", "st")
                            dic.append((normalize(val), f'[{domain}_address]'))
                    elif key == 'name':
                        dic.append((normalize(val), f'[{domain}_name]'))
                        if "b & b" in val:
                            val = val.replace("b & b", "bed and breakfast")
                            dic.append((normalize(val), f'[{domain}_name]'))
                        elif "bed and breakfast" in val:
                            val = val.replace("bed and breakfast", "b & b")
                            dic.append((normalize(val), f'[{domain}_name]'))
                        elif "hotel" in val and 'gonville' not in val:
                            val = val.replace("hotel", "")
                            dic.append((normalize(val), f'[{domain}_name]'))
                        elif "restaurant" in val:
                            val = val.replace("restaurant", "")
                            dic.append((normalize(val), f'[{domain}_name]'))
                    elif key == 'postcode':
                        dic.append((normalize(val), f'[{domain}_postcode]'))
                    elif key == 'phone':
                        dic.append((val, f'[{domain}_phone]'))
                    elif key == 'trainID':
                        dic.append((normalize(val), f'[{domain}_id]'))
                    elif key == 'department':
                        dic.append((normalize(val), f'[{domain}_department]'))

                    elif key == 'area':
                        dic_area.append((normalize(val), '[' + 'value' + '_' + 'area' + ']'))
                    elif key == 'food':
                        dic_food.append((normalize(val), '[' + 'value' + '_' + 'food' + ']'))
                    elif key == 'pricerange':
                        dic_price.append((normalize(val), '[' + 'value' + '_' + 'pricerange' + ']'))
                                    # TODO car type?
        except:
            pass

        if domain == 'hospital':
            dic.extend(
                (
                    (normalize('Hills Rd'), f'[{domain}_address]'),
                    (normalize('Hills Road'), f'[{domain}_address]'),
                    (normalize('CB20QQ'), f'[{domain}_postcode]'),
                    ('01223245151', f'[{domain}_phone]'),
                    ('1223245151', f'[{domain}_phone]'),
                    ('0122324515', f'[{domain}_phone]'),
                    (
                        normalize('Addenbrookes Hospital'),
                        f'[{domain}_' + 'name' + ']',
                    ),
                )
            )
        elif domain == 'police':
            dic.extend(
                (
                    (
                        normalize('Parkside'),
                        '[' + domain + '_' + 'address' + ']',
                    ),
                    (
                        normalize('CB11JG'),
                        '[' + domain + '_' + 'postcode' + ']',
                    ),
                    ('01223358966', '[' + domain + '_' + 'phone' + ']'),
                    ('1223358966', '[' + domain + '_' + 'phone' + ']'),
                    (
                        normalize('Parkside Police Station'),
                        '[' + domain + '_' + 'name' + ']',
                    ),
                )
            )
    with open('data/multi-woz/db/' + 'train' + '_db.json', 'r') as fin:
        db_json = json.load(fin)
    for ent in db_json:
        for key, val in ent.items():
            if key in ['departure', 'destination']:
                dic.append((normalize(val), '[' + 'value' + '_' + 'place' + ']'))

    # add specific values:
    for key in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
        dic.append((normalize(key), '[' + 'value' + '_' + 'day' + ']'))

    # more general values add at the end
    dic.extend(dic_area)
    dic.extend(dic_food)
    dic.extend(dic_price)

    return dic


def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = f' {utt} '.replace(f' {key} ', f' {val} ')
        utt = utt[1:-1]  # why this?

    return utt


def delexicaliseDomain(utt, dictionary, domain):
    for key, val in dictionary:
        if key in [domain, 'value']:
            utt = f' {utt} '.replace(f' {key} ', f' {val} ')
            utt = utt[1:-1]  # why this?

    # go through rest of domain in case we are missing something out?
    for key, val in dictionary:
        utt = f' {utt} '.replace(f' {key} ', f' {val} ')
        utt = utt[1:-1]  # why this?
    return utt

if __name__ == '__main__':
    prepareSlotValuesIndependent()