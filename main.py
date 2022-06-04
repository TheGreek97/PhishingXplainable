# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# regex for URL detection http[s]?:\/\/.*\s
import os
import re


def spam_assassin():
    base_path = 'datasets/raw/SpamAssassin/'
    for folder in ['easy_ham', 'hard_ham']:
        dataset_path = os.path.join(base_path, folder)
        mails = os.listdir(dataset_path)

        file_path = os.path.join(dataset_path, mails[0])
        with open(file_path, mode='r') as m:
            mail = m.read()
            # print(mail)
            # with open('mail.html', 'w') as wr:
            #    wr.writelines(lines)
            extract_features(mail)
        # print(mails[0])


def get_anchors(mail):
    regex = r'<a\s*[^>]*href\s*=\s?[\'|\"][^\'|\"]*[\'|\"]\s*>'
    # regex = r'<a[^>]*>.*<\/a>'
    anchors = re.findall(regex, mail, re.IGNORECASE)
    return anchors


def get_images(mail):
    # regex = r'<img\s*[^>]*src\s*=\s?[\'|\"][^\'|\"]*[\'|\"]\s*/?>'
    regex = r'<img[^>]*>'
    imgs = re.findall(regex, mail, re.IGNORECASE)
    return imgs


def get_buttons(mail):
    regex = r'<button\s*[^>]*onclick\s*=\s?[\'|\"]*[^>]*>.*<\/button>'
    btns = re.findall(regex, mail, re.IGNORECASE)
    return btns


def extract_features(mail):
    links_in_mail = get_anchors(mail)
    images_in_mail = get_images(mail)
    # buttons_in_mail = get_buttons(mail)

    ## Mail body
    # Suspicious Words
    # Image Present
    image_present(images_in_mail)
    # Link Mismatch
    # Special Characters in body
    # Links Present
    links_present(links_in_mail)

    ## Domain Based
    # Age of Domain
    # Expiration
    # Ranking

    ## URL
    for link in links_in_mail:
        href_link = get_link_in_anchor(link)
        if href_link != "":
            # No HTTPS
            https = has_https(href_link)
            # Self-signed HTTPS certificate

            # Special Chars in URL
            spec_chars = special_chars(href_link)
            # Sensitive words in URL

            # IP address
            ip_address = is_ip_address(href_link)
            if not ip_address: # avoid calculating some url features
                # TLD mis-positioned
                tld_mis_pos = is_tld_mispositioned(href_link)
                # Out of position Brand name
                brand_mis_pos = True
                # Number of sub-domains
                num_subdomains = number_subdomains(href_link)
            else:
                tld_mis_pos = False
                brand_mis_pos = False
                num_subdomains = 0
            # URL Length
            # URL is shortened
            # Free domain


def get_link_in_anchor(a_tag):
    href_link = re.search(r'href=[\"|\'][^mailto].*[\"|\']', a_tag)
    if href_link:
        href_link = re.split('=', href_link.group(0))
        return href_link[1]
    else:
        return ""


def image_present(images):
    return len(images) > 0


def links_present(links):
    return len(links) > 0


def has_https(link):
    match = re.match(r'^[\"|\']?https:', link)
    return match is not None


def special_chars(link):
    # TODO: test unicode
    char_dictionary = {
        'slashes': {'reg': r'/'},
        'underscores': {'reg': r'_'},
        'double_slashes': {'reg': r'//'},
        'ats': {'reg': r'@'},
        'dashes': {'reg': r'-'},
        'unicode_chars': {'reg': r'[^\x00-\x7F]+'},
        'digits': {'reg': r'\d'},
    }
    char_counts = {}
    for key, char in char_dictionary.items():
        # print(key, char['reg'])
        matches = re.findall(char['reg'], link)
        char_counts[key] = len(matches)
    return char_counts


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
            if subdomain in common_tld:
                return True
    return False


def number_subdomains(link):
    domain = re.search(r"[^./]+\.[^/]+/", link)
    if domain is None:
        return 0
    else:
        tokens = domain.group(0).split('.')
        return len(tokens[:-2])


if __name__ == '__main__':
    spam_assassin()
