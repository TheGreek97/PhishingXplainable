# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# regex for URL detection http[s]?:\/\/.*\s
import os
import re
from html.parser import HTMLParser

def spam_assassin():
    base_path = 'datasets/raw/SpamAssassin/'
    for folder in ['easy_ham', 'hard_ham']:
        dataset_path = os.path.join(base_path, folder)
        mails = os.listdir(dataset_path)
        # for m in mails:
        #   file_path = os.path.join(dataset_path, m)

        file_path = os.path.join(dataset_path, mails[0])
        with open(file_path, mode='r') as m:
            mail = m.read()
            # TODO separate mails within a single file (divided by -------------------------)
            features = extract_features(mail)
            feature_path = 'datasets/features'
            # with open(os.path.join(feature_path, m), 'w') as output:
            #    output.write(features)

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


def extract_features(mail):
    links_in_mail = get_anchors(mail)
    images_in_mail = get_images(mail)
    # buttons_in_mail = get_buttons(mail)

    ## Mail body
    sus_words_body = suspicious_words_body(mail)  # Suspicious Words
    img_in_body = image_present(images_in_mail)  # Image Present
    special_chars_body = spec_chars_body(mail)  # TODO: Special Characters in body
    links_present_mail = links_present(links_in_mail)  # Links Present
    # TODO Misspelled words

    ## TODO: Domain Based
    age_of_domain = 0  # TODO: Age of Domain
    expiration = 0  # TODO: Expiration
    ranking = 0  # TODO: Ranking

    ## URL
    urls_features = []
    for link in links_in_mail:
        href_link = get_link_in_anchor(link)
        if href_link != "":
            https = has_https(href_link)  # No HTTPS
            self_signed_https = False  # TODO: Self-signed HTTPS certificate
            spec_chars = special_chars(href_link)  # Special Chars in URL
            sensitive_words_url = sensitive_words_in_url(href_link)  # Sensitive words in URL
            ip_address = is_ip_address(href_link)  # IP address
            if not ip_address:  # avoid calculating some url features
                tld_mis_pos = is_tld_mispositioned(href_link)  # TLD mis-positioned
                brand_name_mis_pos = is_brand_name_mispositioned(href_link)  # Out of position Brand name
                num_subdomains = number_subdomains(href_link)  # Number of sub-domains
                url_shortened = is_url_shortened(href_link)  # URL is shortened
            else:  # These 4 features might cause a bias maybe? I want the model to ignore them
                tld_mis_pos = None
                brand_name_mis_pos = None
                num_subdomains = None
                url_shortened = None
            link_mismatch = link_mismatch_a(link)  # Link Mismatch
            url_length = get_url_length(href_link)  # URL Length
            free_domain = False  # TODO: Free domain

            url_feats = {
                "has_https": https,
                "self_signed_https": self_signed_https,
                "spec_chars": spec_chars,
                "sensitive_words_url": sensitive_words_url,
                "tld_mis_pos": tld_mis_pos,
                "brand_name_mis_pos": brand_name_mis_pos,
                "num_subdomains": num_subdomains,
                "url_shortened": url_shortened,
                "link_mismatch": link_mismatch,
                "url_length": url_length,
                "free_domain": free_domain
            }
            urls_features.append(url_feats)

    return {
        "sus_words_body": sus_words_body,
        "img_in_body": img_in_body,
        "special_chars_body": special_chars_body,
        "links_present_mail": links_present_mail,

        "age_of_domain": age_of_domain,
        "expiration": expiration,
        "ranking": ranking,

        "urls_features": urls_features
    }


def suspicious_words_body(mail):
    suspicious_words = ["account", "bank", "outbound", "unsubscribe", "wrote", "pm", "click", "dear", "remove",
                        "contribution", "mailbox", "receive"]
    words_regex = ''
    for i, w in enumerate(suspicious_words):
        words_regex += w
        if i < len(suspicious_words) - 1:
            words_regex += "|"
    match = re.match(words_regex, mail, re.IGNORECASE)
    return match is not None


def image_present(images):
    return len(images) > 0


def spec_chars_body(mail):
    parser = HTMLParser()
    body = parser.feed(mail)
    print(body)
    """
    body = re.search(r'<\s*body\s*>.*<\s*/body\s*>', mail, re.IGNORECASE)
    if body is not None:
        mail = body
    count = 0
    spec_chars_regex = r'[^\x00-\x7F]+'
    matches = re.findall(spec_chars_regex, mail)
    count += len(matches)
    return count"""


def links_present(links):
    return len(links) > 0


def has_https(link):
    match = re.match(r'^[\"|\']?https:', link, re.IGNORECASE)
    return match is not None


def special_chars(link):
    # TODO: test unicode
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
    for key, char in special_chars_dictionary.items():  # This can be optimized by building a single regex
        # print(key, char['reg'])
        matches = re.findall(char['reg'], link)
        char_counts[key] = len(matches)
    return char_counts


def sensitive_words_in_url(link):
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


def is_brand_name_mispositioned(link):
    #TODO how to implement this??
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


if __name__ == '__main__':
    spam_assassin()
