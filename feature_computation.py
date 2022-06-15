# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# regex for URL detection http[s]?:\/\/.*\s
import re
from html.parser import HTMLParser
import utils.ssl_cert
import utils.alexa_rank


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
    regex = r'(https?:\/\/)?(www\.)[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
    iter_reg = re.finditer(regex, mail_text, re.IGNORECASE)
    links = [a.group(0) for a in iter_reg]
    return links


def extract_features(mail):
    anchors_in_mail = get_anchors(mail)
    images_in_mail = get_images(mail)
    # buttons_in_mail = get_buttons(mail)
    links_plain_text = []
    email_is_html = True

    if len(anchors_in_mail) < 1:
        links_plain_text = get_links_plain_text(mail)
        if len(links_plain_text) < 1:
            # print(mail)
            return False
        else:
            email_is_html = False

    ## Mail body
    sus_words_body = suspicious_words_body(mail)  # Suspicious Words
    img_in_body = image_present(images_in_mail)  # Image Present
    special_chars_body = spec_chars_body(mail)  # TODO: Special Characters in body
    links_present_mail = links_present(mail)  # Links Present
    # TODO Misspelled words

    if email_is_html:
        ## URL
        urls_features = []
        for link in anchors_in_mail:
            href_link = get_link_in_anchor(link)
            if href_link != "":
                url_feats = get_url_features(href_link, link)
                urls_features.append(url_feats)
    else:
        ## URL
        urls_features = []
        for link in links_plain_text:
            if link != "":
                url_feats = get_url_features(link)
                urls_features.append(url_feats)

    return {
        "sus_words_body": sus_words_body,
        "img_in_body": img_in_body,
        "special_chars_body": special_chars_body,
        "links_present_mail": links_present_mail,

        "urls_features": urls_features
    }


def get_url_features(link, visible_link=""):
    hostname = get_hostname(link)
    ##  Domain Based
    age_of_domain = 0  # TODO: Age of Domain
    expiration = 0  # TODO: Expiration
    ranking = 0  # = utils.alexa_rank.getRank(hostname)

    https = has_https(link)  # No HTTPS
    self_signed_https = False  # = self_signed_HTTPS(link, hostname) Non-valid SSL certificate
    spec_chars = special_chars(link)  # Special Chars in URL
    sensitive_words_url = sensitive_words_in_url(link)  # Sensitive words in URL
    ip_address = is_ip_address(link)  # IP address
    if not ip_address:  # avoid calculating some url features
        tld_mis_pos = is_tld_mispositioned(link)  # TLD mis-positioned
        brand_name_mis_pos = is_brand_name_mispositioned(link)  # Out of position Brand name
        num_subdomains = number_subdomains(link)  # Number of sub-domains
        url_shortened = is_url_shortened(link)  # URL is shortened
    else:  # These 4 features might cause a bias maybe? I want the model to ignore them
        tld_mis_pos = None
        brand_name_mis_pos = None
        num_subdomains = None
        url_shortened = None
    if visible_link != "":
        link_mismatch = link_mismatch_a(visible_link)  # Link Mismatch
    else:
        link_mismatch = False
    url_length = get_url_length(link)  # URL Length
    free_domain = False  # TODO: Free domain

    url_features = {
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
        "free_domain": free_domain,
        "age_of_domain": age_of_domain,
        "expiration": expiration,
        "ranking": ranking
    }

    return url_features


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
    print (mail)
    # SEPARATE THE BODY FROM THE EMAIL (Not with <body> tag, since a lot of the mails are in plain-text, maybe something like \n\n?)
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


def self_signed_HTTPS(url, hostname):
    # print (url)
    if has_https(url):
        port = 443
        cert = utils.ssl_cert.get_certificate(hostname, port)
        valid_cert = utils.ssl_cert.verify_cert(cert, hostname)
        return not valid_cert
    else:
        return False
