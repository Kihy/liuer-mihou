#Check if cython code has been compiled
import os
import subprocess
import copy
#Import dependencies
import after_image.net_stat as ns
import csv
import numpy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess
import pickle
from scapy.layers.http import *
from scapy.layers.inet import TCP


#Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
class FE:
    def __init__(self,file_path, parse_type, limit=None, nstat=None, dummy_nstat=None, force_tsv=True, log_file=None):
        self.path = file_path
        self.limit = limit
        self.parse_type = None #unknown
        self.curPacketIndx = 0
        self.input_stream=None
        self.parse_type=parse_type
        self.force_tsv=force_tsv


        if self.parse_type=="tsv":
            # prepare tsv file
            if self.path.endswith("pcap"):
                if not os.path.isfile(self.path+".tsv") or self.force_tsv:
                    self.pcap2tsv_with_tshark()
                self.path+=".tsv"
            self.input_stream=csv.reader(open(self.path, "r"),delimiter="\t")
            #skip header
            next(self.input_stream)

        else:
            ### Prep pcap ##
            print("Reading PCAP file via Scapy...")

            # self.scapyin = rdpcap(self.path, count=self.max_pkt)
            self.input_stream = PcapReader(self.path)

        ### Prep Feature extractor (AfterImage) ###
        self.maxHost = 100000000000
        self.maxSess = 100000000000

        if nstat is not None:
            self.nstat=nstat
        else:
            self.nstat = ns.netStat(np.nan, self.maxHost, self.maxSess, log_path=log_file)

        # if dummy_nstat is not None:
        #     self.dummy_nstat=dummy_nstat
        # else:
        #     self.dummy_nstat=ns.netStat(np.nan, self.maxHost, self.maxSess)

    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst -e ip.proto -e llc.type"
        if self.limit is not None:
            options=" -c "+ self.limit
        else:
            options=""
        cmd =  'tshark -r '+ self.path +' -T fields '+ fields + options +' -E header=y -E occurrence=f > '+self.path+".tsv"
        subprocess.call(cmd,shell=True)
        print("tshark parsing complete. File saved as: "+self.path +".tsv")


    def get_next_vector(self):
        if self.parse_type=="tsv":
            return self.get_next_vector_tsv()
        elif self.parse_type=="scapy":
            return self.get_next_vector_scapy()

    def get_nstat(self):
        return self.nstat #, self.dummy_nstat

    def get_next_vector_tsv(self):
        row = next(self.input_stream)
        IPtype = np.nan
        packet=row
        if row[-1] == "":
            self.curPacketIndx = self.curPacketIndx + 1
            return []
        timestamp = row[0]
        framelen = row[1]
        srcIP = ''
        dstIP = ''
        if row[4] != '':  # IPv4
            srcIP = row[4]
            dstIP = row[5]
            IPtype = 0
        elif row[17] != '':  # ipv6
            srcIP = row[17]
            dstIP = row[18]
            IPtype = 1
        srcproto = row[6] + row[
            8]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
        dstproto = row[7] + row[9]  # UDP or TCP port
        srcMAC = row[2]
        dstMAC = row[3]
        if srcproto == '':  # it's a L2/L1 level protocol
            if row[12] != '':  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = row[14]  # src IP (ARP)
                dstIP = row[16]  # dst IP (ARP)
                IPtype = 0
            elif row[10] != '':  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                srcIP = row[2]  # src MAC
                dstIP = row[3]  # dst MAC

        return  [IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, float(timestamp), int(framelen)]


    def get_next_vector_scapy(self):
        pkt_tuple = self.input_stream.read_packet()
        packet, pkt_metadata=pkt_tuple[0],pkt_tuple[1]

        # only process IP packets,
        if not (packet.haslayer(IP) or packet.haslayer(IPv6) or packet.haslayer(ARP)):
            return [], packet
        # packet.show2()
        # print(dir(packet))

        timestamp = packet.time
        framelen = len(packet)
        if packet.haslayer(IP):  # IPv4
            srcIP = packet[IP].src
            dstIP = packet[IP].dst
            IPtype = 0
        elif packet.haslayer(IPv6):  # ipv6
            srcIP = packet[IPv6].src
            dstIP = packet[IPv6].dst
            IPtype = 1
        else:
            srcIP = ''
            dstIP = ''

        if packet.haslayer(TCP):
            srcproto = str(packet[TCP].sport)
            dstproto = str(packet[TCP].dport)
        elif packet.haslayer(UDP):
            srcproto = str(packet[UDP].sport)
            dstproto = str(packet[UDP].dport)
        else:
            srcproto = ''
            dstproto = ''

        if packet.haslayer(ARP):
            srcMAC=packet[ARP].hwsrc
            dstMAC=packet[ARP].hwdst
        else:
            srcMAC = packet.src
            dstMAC = packet.dst

        if srcproto == '':  # it's a L2/L1 level protocol
            if packet.haslayer(ARP):  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = packet[ARP].psrc  # src IP (ARP)
                dstIP = packet[ARP].pdst  # dst IP (ARP)
                IPtype = 0
            elif packet.haslayer(ICMP):  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                srcIP = packet.src  # src MAC
                dstIP = packet.dst  # dst MAC

        return [IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, float(timestamp), int(framelen)], packet

    # def save_nstat_state(self):
    #     f=open('tmp_nstat.txt', 'wb')
    #     pickle.dump( obj=self.nstat,file=f)
    #
    # def roll_back(self):
    #     """Roll back dummy to nstat"""
    #
    #     self.dummy_nstat = pickle.load(open('tmp_nstat.txt', 'rb'))


    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())
