import json

import requests
# from bs4 import BeautifulSoup
import re
import validators


opr_url = 'https://openpagerank.com/api/v1.0/getPageRank'


def is_valid_domain(site_name):
    if validators.domain(site_name):
        return True
    else:
        return False


def getRank(url):
    url = opr_url + "?domains[]=" + url + ""
    response = requests.get(url, headers={"API-OPR": "gs48s8sgko4cgkkcogkc8gsg0s0o40s8kg84wkss"})
    response = json.loads(response.text)
    if response["status_code"] == 200:
        rank = response["response"][0]["rank"]
        if rank is not None:
            global_rank = int(rank)
        else:
            global_rank = 999999999
    else:
        global_rank = 999999999
    return global_rank
