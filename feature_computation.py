# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# regex for URL detection http[s]?:\/\/.*\s
import re
import math
import utils.ssl_cert
import utils.alexa_rank
from html.parser import HTMLParser
from datetime import datetime
from whois import whois
from spellchecker import SpellChecker


def extract_features(mail):
    anchors_in_mail = get_anchors(mail)
    images_in_mail = get_images(mail)
    links_plain_text = get_links_plain_text(mail)
    email_is_html = True

    if len(anchors_in_mail) < 1:  # If there's no link in the mail, don't compute the features
        if len(links_plain_text) < 1:
            # print(mail)
            return False
        else:
            email_is_html = False

    ## Mail body
    sus_words_body = suspicious_words_body(mail)  # Suspicious Words
    img_in_body = image_present(images_in_mail)  # Image Present
    special_chars_body = spec_chars_body(mail)  # Special Characters in body
    links_present_mail = len(links_plain_text)  # Links Present
    no_misspelled_words = misspelled_words(mail)  # Misspelled words

    url_features = None
    if email_is_html:
        ## URL
        max_score_phish = -1
        for link in anchors_in_mail:
            href_link = get_link_in_anchor(link)
            if href_link != "":
                url_feats, new_score = get_url_features(href_link, visible_link=link, max_score_phish=max_score_phish)
                if new_score > max_score_phish:
                    max_score_phish = new_score
                    url_features = url_feats  # urls_features.append(url_feats)
    else:
        ## URL
        max_score_phish = -1
        for link in links_plain_text:
            if link != "":
                url_feats, new_score = get_url_features(link, max_score_phish=max_score_phish)
                if new_score > max_score_phish:
                    max_score_phish = new_score
                    url_features = url_feats

    if url_features is not None:
        body_features = {
            "sus_words_body": sus_words_body,
            "img_in_body": img_in_body,
            "special_chars_body": special_chars_body,
            "links_present_mail": links_present_mail,
            "no_misspelled_words": no_misspelled_words
        }
        return body_features | url_features  # merge the 2 dictionaries
    else:
        return False


def get_anchors(mail):
    anchors_regex = r'<a[^>]*href\s*=\s*((\'[^\']*\')|(\"[^"]*\")).*>[^<]*<\s*\/a\s*>'
    anchors = re.finditer(anchors_regex, mail, re.IGNORECASE)
    tags = [a.group(0) for a in anchors]
    return tags


def get_images(mail):
    # regex = r'<img\s*[^>]*src\s*=\s?[\'|\"][^\'|\"]*[\'|\"]\s*/?>'
    regex = r'<img[^>]*>'
    imgs = re.findall(regex, mail, re.IGNORECASE)
    return imgs


def get_buttons(mail):
    regex = r'<button\s*[^>]*onclick\s*=\s?[\'|\"]*[^>]*>.*<\/button>'
    btns = re.findall(regex, mail, re.IGNORECASE)
    return btns


def get_link_in_anchor(a_tag):
    href_link = re.search(r'href\s*=\s*((\'[^\']*\')|(\"[^"]*\"))', a_tag,  re.IGNORECASE)
    # href=[\"|\'][^mailto].*[\"|\']
    if href_link:
        href_link = re.split('=', href_link.group(0))
        link = href_link[1]
        is_mailto = re.search(r'mailto:', link, re.IGNORECASE)
        if is_mailto is None:
            return link
    return ""


def get_hostname(url):
    regex = r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)'
    match = re.search(regex, url, re.IGNORECASE)
    if match and match.group(1):
        hostname = match.group(1)
        return hostname
    else:
        return ""


def get_links_plain_text (mail_text):
    # regex = r'(https?:\/\/)?(www\.)[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
    regex = r'((https?:\/\/)|(www\.))[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
    iter_reg = re.finditer(regex, mail_text, re.IGNORECASE)
    img_extension = ('.gif', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.eps')
    links = [a.group(0) for a in iter_reg if not a.group(0).endswith(img_extension)]
    return links


def get_url_features(link, visible_link="", max_score_phish=0):
    score = 0
    hostname = get_hostname(link)

    no_https = not has_https(link)  # No HTTPS
    score = score + 1 if no_https else score

    self_signed_https = self_signed_HTTPS(link, hostname)  # Non-valid SSL certificate
    score = score + 1 if self_signed_https else score

    spec_chars, no_spec_chars_url = special_chars(link)  # Special Chars in URL
    increase = math.floor(no_spec_chars_url / 5)  # increase the score by 1 for each 4 special chars in the URL
    increase = increase if increase < 5 else 5  # (max increase is 4)
    score = score + increase

    sensitive_words_url = no_sensitive_words_in_url(link)  # Sensitive words in URL
    score = score + sensitive_words_url

    ip_address = is_ip_address(link)  # IP address
    score = score + 3 if ip_address else score

    if not ip_address:  # avoid calculating some url features
        url_shortened = is_url_shortened(link)  # URL is shortened
        score = score + 2 if url_shortened else score
        if not url_shortened:
            tld_mis_pos = is_tld_mispositioned(link)  # TLD mis-positioned
            score = score + 1 if tld_mis_pos else score
            num_subdomains = number_subdomains(link)  # Number of sub-domains
            score = score + 1 if num_subdomains > 1 else score
        else:
            tld_mis_pos = False
            num_subdomains = 1
    else:  # The features that are set to default values might cause a bias? I want the model to ignore them
        tld_mis_pos = None
        num_subdomains = None
        url_shortened = None

    if visible_link != "":
        link_mismatch = link_mismatch_a(visible_link)  # Link Mismatch
        score = score + 1 if link_mismatch else score
    else:
        link_mismatch = False

    url_length = get_url_length(link)  # URL Length
    increase = math.floor(url_length / 50)  # increase the score by 1 for each 50 chars in the URL
    increase = increase if increase < 5 else 5  # (max increase is 4)
    score = score + increase

    # Domain-Based
    if score > max_score_phish - 2:  # to avoid computing too many costly features
        age_of_domain, expiration = whois_info(link)  # Age of Domain (in days), Days until Expiration
        ranking = utils.alexa_rank.getRank(hostname)
    else:  # dummy values
        age_of_domain = 9999999
        expiration = 99999999
        ranking = 20000
    score = score + 1 if age_of_domain < 200 else score  # increase score if the domain is less than 200 days old
    score = score + 1 if expiration < 100 else score  # increase score if domain is expiring in less than 100 days

    score = score + 1 if ranking > 20000 else score  # increase score if the Alexa ranking is more than 20k

    url_features = {
        "url_no_https": no_https,
        "url_self_signed_https": self_signed_https,
        "url_spec_chars": spec_chars,
        "url_sensitive_words": sensitive_words_url,
        "url_tld_mis_pos": tld_mis_pos,
        "url_num_subdomains": num_subdomains,
        "url_shortened": url_shortened,
        "url_link_mismatch": link_mismatch,
        "url_length": url_length,
        "url_age_of_domain": age_of_domain,
        "url_domain_expiration": expiration,
        "url_ranking": ranking,
        "url_ip_address": ip_address
    }
    return url_features, score


def suspicious_words_body(mail):
    suspicious_words = ["account", "bank", "outbound", "unsubscribe", "wrote", "pm", "click", "dear", "remove",
                        "contribution", "mailbox", "receive"]
    words_regex = ''
    for i, w in enumerate(suspicious_words):
        words_regex += w
        if i < len(suspicious_words) - 1:
            words_regex += "|"
    match = re.findall(words_regex, mail, re.IGNORECASE)
    count = len(match)
    return count


def image_present(images):
    return len(images) > 0


def misspelled_words(mail):
    spell = SpellChecker()
    body = get_mail_body(mail)
    body = re.sub(r'([!\?,\.\(\)><#£$€\[\]@:;\"\-\/_\\+]|^\x00-\x7F)', " ", body)
    words_in_mail = re.split(r'\s+', body)
    unknown_words = spell.unknown(words_in_mail)
    for w in ['', 'www', 'http', 'https']:  # remove some words
        try:
            unknown_words.remove(w)
        except KeyError:
            continue
    return len(unknown_words)


def spec_chars_body(mail):
    body = get_mail_body(mail)
    matches_spec_chars = re.findall(r'[.,!?@\\:£$€\/\-;\*%\+_]', body)
    matches_unicode_chars = re.findall(r'[^\x00-\x7F]', body)
    count = len(matches_spec_chars) + len(matches_unicode_chars)
    return count


def links_present(links):
    return len(links)


def whois_info(url):
    url = url.lower()
    try:
        w = whois(url)
        if w is not None:
            exp_date = w.expiration_date
            creat_date = w.creation_date  # datetime.strptime(w.creation_date, "%Y-%m-%d %H:%M:%S")
            if exp_date is not None:
                exp_date = exp_date - datetime.now()
                exp_date = exp_date.days
            else:
                exp_date = 0
            if creat_date is None:
                creat_date = datetime.now()
            age_of_domain = datetime.now() - creat_date
            age_of_domain = age_of_domain.days
        else:
            age_of_domain = 0
            exp_date = 0
        return age_of_domain, exp_date
    except:
        return 0, 0


def has_https(link):
    match = re.match(r'^[\"|\']?https:', link, re.IGNORECASE)
    return match is not None


def special_chars(link):
    char_counts = {}
    special_chars_dictionary = {
        'slashes': {'reg': r'/'},
        'underscores': {'reg': r'_'},
        'double_slashes': {'reg': r'//'},
        'ats': {'reg': r'@'},
        'dashes': {'reg': r'-'},
        'unicode_chars': {'reg': r'[^\x00-\x7F]+'},
        'digits': {'reg': r'\d'},
    }
    tot_number_spec_chars = 0
    for key, char in special_chars_dictionary.items():  # This can be optimized by building a single regex
        # print(key, char['reg'])
        matches = re.findall(char['reg'], link)
        char_counts[key] = len(matches)
        tot_number_spec_chars += len(matches)
    return char_counts, tot_number_spec_chars


def no_sensitive_words_in_url(link):
    sensitive_words = ["secure", "webscr", "login", "account", "ebay", "signin", "banking", "confirm"]
    words_regex = ''
    for i, w in enumerate (sensitive_words):
        words_regex += w
        if i < len(sensitive_words)-1:
            words_regex += "|"
    matches = re.findall(words_regex, link, re.IGNORECASE)
    return len(matches)


def is_ip_address(link):
    match = re.match(r'([\d]{1,3}\.){3}[\d]{1,3}', link)
    return match is not None


def is_tld_mispositioned(link):
    domain = re.search(r"[^./]+\.[^/]+/", link)
    # Top 20 TLD https://www.seobythesea.com/2006/01/googles-most-popular-and-least-popular-top-level-domains/ + other 5
    common_tld = ['.com', '.org', '.edu', '.gov', '.uk', '.net', '.ca', '.de', '.jp', '.fr', '.au', '.us', 'ru', '.ch',
                  '.it', '.nl', 'se', '.no', '.es', '.mil', '.info', '.tk', '.cn', 'xyz', 'top']
    if domain is not None:
        tokens = domain.group(0).split('.')[:-2]
        for subdomain in tokens:
            if subdomain.lower() in common_tld:
                return True
    return False


def number_subdomains(link):
    domain = re.search(r"[^./]+\.[^/]+/", link)
    if domain is None:
        return 0
    else:
        tokens = domain.group(0).split('.')
        return len(tokens[:-2])


def link_mismatch_a(link):
    href = re.search(r"href[\s]*=[\s]*(([\"][^\"]*[\"])|([\'][^\']*[\']))", link, re.IGNORECASE)
    visible_link = re.search(r">[^>]+<", link)

    if href is None or visible_link is None:
        return False
    else:
        href = href.group(0).lower()
        href = href.split('href')[1]
        href = href.strip('="\'')
        visible_link = visible_link.group(0)
        visible_link = visible_link.strip('<>')
        return href != visible_link


def is_url_shortened(link):
    # Top 12 used URL Shortener services https://blog.hootsuite.com/what-are-url-shorteners/
    services = ["ow.ly", "t.co", "bit.ly", 'tinyurl', 'tiny.cc', 'bit.do', 'shorte.st', 'cut.ly']
    words_regex = ''
    for i, w in enumerate(services):
        words_regex += w
        if i < len(services) - 1:
            words_regex += "|"
    match = re.match(words_regex, link, re.IGNORECASE)
    return match is not None


def get_url_length(link):
    return len(link)


def self_signed_HTTPS(url, hostname):
    # print (url)
    if has_https(url):
        port = 443
        cert = utils.ssl_cert.get_certificate(hostname, port)
        valid_cert = utils.ssl_cert.verify_cert(cert, hostname)
        return not valid_cert
    else:
        return False


def get_mail_body(mail):
    parser = HTMLParser()
    body = parser.feed(mail)
    if body is None:
        body = mail
    splits = re.split(r"X-FileName:.*\n", body, 2)
    if len(splits) > 1:
        body = splits[1]
    return body


def compute_links_in_mail_feature(mail):
    anchors_in_mail = get_anchors(mail)
    links_plain_text = get_links_plain_text(mail)
    if len(anchors_in_mail) < 1:  # If there's no link in the mail, don't compute the features
        if len(links_plain_text) < 1:
            return False
    return len(links_plain_text)
