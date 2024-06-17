# -*- encoding: utf-8 -*-
# requires a recent enough python with idna support in socket
# pyopenssl, cryptography and idna
import ssl
from ssl import SSLContext
import certifi
import os

from datetime import datetime
from OpenSSL import SSL
from cryptography import x509
from cryptography.x509.oid import NameOID
import idna
#from backports.ssl_match_hostname import match_hostname, CertificateError

import concurrent.futures
from socket import socket, SOCK_STREAM, AF_INET
from collections import namedtuple

HostInfo = namedtuple(field_names='cert hostname peername', typename='HostInfo')

HOSTS = [
    ('damjan.softver.org.mk', 443),
    ('expired.badssl.com', 443),
    ('wrong.host.badssl.com', 443),
    ('ca.ocsr.nl', 443),
    ('faß.de', 443),
    ('самодеј.мкд', 443),
]


def has_expired(cert):
    exp_date = cert["notAfter"][:-4]
    # print (exp_date)
    exp_date = datetime.strptime(exp_date, '%b %d %H:%M:%S %Y')
    today = datetime.now()
    return today >= exp_date


def verify_cert(cert, hostname):
    # verify notAfter/notBefore, CA trusted, servername/sni/hostname
    # service_identity.pyopenssl.verify_hostname(client_ssl, hostname)
    # issuer
    if cert == {}:
        return False
    expired = has_expired(cert)
    """try:
        match_hostname(cert, hostname)
        return True and not expired
    except CertificateError:
        return False"""
    return not expired


def get_certificate(hostname, port):
    # hostname_idna = idna.encode(hostname)

    try:
        # print(hostname)
        ctx = SSLContext(ssl.PROTOCOL_TLSv1_2)  # most compatible
        # ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(
            cafile=os.path.relpath(certifi.where()),
            capath=None,
            cadata=None)

        sock = socket(AF_INET, SOCK_STREAM)
        ssl_sock = ctx.wrap_socket(sock)  # , server_hostname=hostname)
        ssl_sock.connect((hostname, port))
        # sock.connect((hostname, port))
        # peer_name = sock.getpeername()
        # ctx = SSL.Context(SSL.SSLv23_METHOD) # most compatible
        # ctx.check_hostname = False
        # ctx.verify_mode = SSL.VERIFY_NONE
        # ssl_sock = ssl.wrap_socket(sock, ssl_version=ssl.PROTOCOL_SSLv3,
        #                            cert_reqs=ssl.CERT_REQUIRED, ca_certs=...)
        cert = None
        """try:
            cert = ssl_sock.getpeercert()
            # print(cert)
        except CertificateError:
            cert = None"""
        ssl_sock.unwrap()
        ssl_sock.close()
        sock.close()
        return cert
        """
        sock_ssl = SSL.Connection(ctx, sock)
        sock_ssl.set_connect_state()
        sock_ssl.set_tlsext_host_name(hostname_idna)
        sock_ssl.do_handshake()
        cert = sock_ssl.get_peer_certificate()
        crypto_cert = cert.to_cryptography()
        sock_ssl.close()
        sock.close()
        return HostInfo(cert=crypto_cert, peername=peer_name, hostname=hostname)"""
    except:
        sock.close()
        return {}


def get_alt_names(cert):
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        return ext.value.get_values_for_type(x509.DNSName)
    except x509.ExtensionNotFound:
        return None


def get_common_name(cert):
    try:
        names = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        return names[0].value
    except x509.ExtensionNotFound:
        return None


def get_issuer(cert):
    try:
        names = cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
        return names[0].value
    except x509.ExtensionNotFound:
        return None


def print_basic_info(hostinfo):
    s = '''» {hostname} « … {peername}
    \tcommonName: {commonname}
    \tSAN: {SAN}
    \tissuer: {issuer}
    \tnotBefore: {notbefore}
    \tnotAfter:  {notafter}
    '''.format(
            hostname=hostinfo.hostname,
            peername=hostinfo.peername,
            commonname=get_common_name(hostinfo.cert),
            SAN=get_alt_names(hostinfo.cert),
            issuer=get_issuer(hostinfo.cert),
            notbefore=hostinfo.cert.not_valid_before,
            notafter=hostinfo.cert.not_valid_after
    )
    print(s)


"""
def check_it_out(hostname, port):
    hostinfo = get_certificate(hostname, port)
    print_basic_info(hostinfo)



if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:
        for hostinfo in e.map(lambda x: get_certificate(x[0], x[1]), HOSTS):
            print_basic_info(hostinfo)
"""
