import numpy as np
## Prep AfterImage cython package
import os
import subprocess
import pyximport
pyximport.install()
import after_image.after_image as af
from after_image.after_image import IncStat1D, IncStat2D
from pprint import pformat
import copy
#import AfterImage_NDSS as af

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    #Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(self, Lambdas = np.nan, HostLimit=255,HostSimplexLimit=1000, log_path=None):
        #Lambdas
        if np.isnan(Lambdas):
            self.Lambdas = [5,3,1,.1,.01]
        else:
            self.Lambdas = Lambdas
        self.clean_up_round=100
        # number of pkts updated
        self.num_updated=0

        self.prev_pkt_time=0

        #cutoffweight for cleaning
        self.cutoffWeight=1e-6

        #HT Limits
        self.HostLimit = HostLimit
        self.SessionLimit = HostSimplexLimit*self.HostLimit*self.HostLimit #*2 since each dual creates 2 entries in memory
        self.MAC_HostLimit = self.HostLimit*10

        #HTs
        self.HT_jit = af.IncStatDB("HT_jit",limit=self.HostLimit*self.HostLimit)#H-H Jitter Stats
        self.HT_MI = af.IncStatDB("HT_MI",limit=self.MAC_HostLimit)#MAC-IP relationships
        self.HT_H = af.IncStatDB("HT_H",limit=self.HostLimit) #Source Host BW Stats
        self.HT_Hp = af.IncStatDB("HT_Hp",limit=self.SessionLimit)#Source Host BW Stats

        print("log_path:", log_path)
        if log_path is not None:
            print("netstat log_path",log_path)
            self.log_file=open(log_path,"w")
        else:
            self.log_file =None

    def getHT(self):
        return {"HT_jit":self.HT_jit,
        "HT_MI" :self.HT_MI,
        "HT_H":self.HT_H,
        "HT_Hp":self.HT_Hp,
        "num_updated":self.num_updated}

    def setHT(self, HT_dict):
        self.HT_jit=HT_dict["HT_jit"]
        self.HT_MI=HT_dict["HT_MI"]
        self.HT_H=HT_dict["HT_H"]
        self.HT_Hp=HT_dict["HT_Hp"]
        self.num_updated=HT_dict["num_updated"]

    def __repr__(self):
        return "HT_jit"+pformat(self.HT_jit, indent=2)+"\nHT_MI"+pformat(self.HT_MI, indent=2)+"\nHT_H"+pformat(self.HT_H, indent=2)+"\nHT_Hp"+pformat(self.HT_Hp, indent=2)

    def findDirection(self,IPtype,srcIP,dstIP,eth_src,eth_dst): #cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        if IPtype==0: #is IPv4
            lstP = srcIP.rfind('.')
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind('.')
            dst_subnet = dstIP[0:lstP:]
        elif IPtype==1: #is IPv6
            src_subnet = srcIP[0:round(len(srcIP)/2):]
            dst_subnet = dstIP[0:round(len(dstIP)/2):]
        else: #no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def get_records(self,IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize):
        db={}
        db["HT_MI"]=[]
        db["HHstat"] = [[]for i in range(len(self.Lambdas))]
        db["HHstat_cov"]=[]
        db["HHstat_jit"]=[]
        db["HpHpstat"] = [[]for i in range(len(self.Lambdas))]
        db["HpHpstat_cov"]=[]
        db["num_updated"]=self.num_updated

        for i in range(len(self.Lambdas)):
            inc_stat=self.HT_MI.register("{}_{}".format(srcMAC,srcIP), i)
            db["HT_MI"].append(inc_stat)

            inc_stat=self.HT_H.register(srcIP,i)
            db["HHstat"][i].append(inc_stat)

            inc_stat=self.HT_H.register(dstIP,i)
            db["HHstat"][i].append(inc_stat)

            inc_cov=self.HT_H.register_cov(srcIP, dstIP,i)
            db["HHstat_cov"].append(inc_cov)

            inc_stat=self.HT_jit.register("{}_{}".format(srcIP,dstIP),i,isTypeDiff=True)
            db["HHstat_jit"].append(inc_stat)


            if srcProtocol == 'arp':

                inc_stat= self.HT_Hp.register(srcMAC,i)
                db["HpHpstat"][i].append(inc_stat)

                inc_stat= self.HT_Hp.register(dstMAC,i)
                db["HpHpstat"][i].append(inc_stat)

                inc_cov=self.HT_Hp.register_cov(srcMAC, dstMAC, i)
                db["HpHpstat_cov"].append(inc_cov)
            else:  # some other protocol (e.g. TCP/UDP)
                inc_stat= self.HT_Hp.register("{}_{}".format(srcIP, srcProtocol), i)
                db["HpHpstat"][i].append(inc_stat)

                inc_stat= self.HT_Hp.register("{}_{}".format(dstIP , dstProtocol), i)
                db["HpHpstat"][i].append(inc_stat)
                inc_cov=self.HT_Hp.register_cov("{}_{}".format(srcIP, srcProtocol), "{}_{}".format(dstIP , dstProtocol),i)
                db["HpHpstat_cov"].append(inc_cov)
        return db

    def update_dummy_db(self, IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize, db,verbose=False):
        # update
        record=np.zeros(len(self.getNetStatHeaders()))

        for i in range(len(self.Lambdas)):
            if db["HT_MI"][i] is None:
                inc_stat=IncStat1D(self.Lambdas[i], "{}_{}".format(srcMAC,srcIP), timestamp)
                db["HT_MI"][i]=inc_stat
            db["HT_MI"][i].insert(datagramSize,timestamp)
            record[i*3:(i+1)*3]=db["HT_MI"][i].all_stats_1D()

            if db["HHstat"][i][0] is None:
                inc_stat=IncStat1D(self.Lambdas[i], srcIP, timestamp)
                db["HHstat"][i][0]=inc_stat
            db["HHstat"][i][0].insert(datagramSize,timestamp)

            if db["HHstat"][i][1] is None:
                inc_stat=IncStat1D(self.Lambdas[i], dstIP, timestamp)
                db["HHstat"][i][1]=inc_stat

            if db["HHstat_cov"][i] is None:
                incS1 = db["HHstat"][i][0]
                incS2 = db["HHstat"][i][1]

                inc_cov = IncStat2D(incS1,incS2,timestamp)
                incS1.add_cov(dstIP)
                incS2.add_cov(srcIP)
                db["HHstat_cov"][i]=inc_cov
            db["HHstat_cov"][i].update_cov(srcIP, datagramSize,timestamp)
            record[i*7+15:(i+1)*7+15]=db["HHstat"][i][0].all_stats_1D()+db["HHstat_cov"][i].get_stats2()



            if db["HHstat_jit"][i] is None:
                inc_stat=IncStat1D(self.Lambdas[i], "{}_{}".format(srcIP,dstIP), timestamp,isTypeDiff=True)
                db["HHstat_jit"][i]=inc_stat
            db["HHstat_jit"][i].insert(0,timestamp)
            record[(i*3)+50:((i+1)*3+50)]+=db["HHstat_jit"][i].all_stats_1D()


            if srcProtocol == 'arp':

                if db["HpHpstat"][i][0] is None:
                    inc_stat= IncStat1D(self.Lambdas[i], srcMAC, timestamp)
                    db["HpHpstat"][i][0]=inc_stat
                db["HpHpstat"][i][0].insert(datagramSize,timestamp)

                if db["HpHpstat"][i][1] is None:
                    inc_stat=IncStat1D(self.Lambdas[i], dstMAC, timestamp)
                    db["HpHpstat"][i][1]=inc_stat


                if db["HpHpstat_cov"][i] is None:
                    incS1 = db["HpHpstat"][i][0]
                    incS2 = db["HpHpstat"][i][1]

                    inc_cov = IncStat2D(incS1,incS2,timestamp)
                    incS1.add_cov(dstMAC)
                    incS2.add_cov(srcMAC)
                    db["HpHpstat_cov"][i]=inc_cov
                db["HpHpstat_cov"][i].update_cov(srcMAC,datagramSize,timestamp)

                record[(i*7)+65:((i+1)*7)+65]=db["HpHpstat"][i][0].all_stats_1D() + db["HpHpstat_cov"][i].get_stats2()
            else:

                if db["HpHpstat"][i][0] is None:
                    inc_stat=IncStat1D(self.Lambdas[i], "{}_{}".format(srcIP, srcProtocol), timestamp)
                    db["HpHpstat"][i][0] = inc_stat
                db["HpHpstat"][i][0].insert(datagramSize, timestamp)

                if db["HpHpstat"][i][1] is None:
                    inc_stat=IncStat1D(self.Lambdas[i], "{}_{}".format(dstIP , dstProtocol), timestamp)
                    db["HpHpstat"][i][1] = inc_stat

                if db["HpHpstat_cov"][i] is None:
                    incS1 = db["HpHpstat"][i][0]
                    incS2 = db["HpHpstat"][i][1]

                    inc_cov = IncStat2D(incS1,incS2,timestamp)
                    incS1.add_cov("{}_{}".format(dstIP , dstProtocol))
                    incS2.add_cov("{}_{}".format(srcIP, srcProtocol))
                    db["HpHpstat_cov"][i]=inc_cov

                db["HpHpstat_cov"][i].update_cov("{}_{}".format(srcIP, srcProtocol),datagramSize,timestamp)
                record[(i*7)+65:((i+1)*7)+65]=db["HpHpstat"][i][0].all_stats_1D() + db["HpHpstat_cov"][i].get_stats2()

        #mimic cleaning behavior
        db["num_updated"]+=1
        if db["num_updated"]%self.clean_up_round==0:
            if verbose:
                print(db["num_updated"])
                print("dummy clean timestamp",timestamp)
            self.clean_dummy_old_records(self.cutoffWeight, timestamp, db, "HT_MI", verbose)
            self.clean_dummy_old_records(self.cutoffWeight, timestamp, db, "HHstat", verbose)
            self.clean_dummy_old_records(self.cutoffWeight, timestamp, db,"HHstat_jit", verbose)
            self.clean_dummy_old_records(self.cutoffWeight, timestamp, db, "HpHpstat", verbose)
            if verbose:
                print(db)

        return record

    def clean_dummy_old_records(self,cutoffWeight,curTime,db,ht,verbose=False):
        for i in range(len(self.Lambdas)):
            if not isinstance(db[ht][i], list):
                inc_stat=db[ht][i]
                inc_stat.processDecay(curTime)
                if inc_stat.weight() < cutoffWeight:
                    # print("cleaning dummy record",int_stat)
                    db[ht][i]=None
            else:
                inc_stat=db[ht][i][0]
                inc_stat.processDecay(curTime)
                inc_stat2=db[ht][i][1]
                inc_stat2.processDecay(curTime)

                remove_flag=False
                if inc_stat.weight() < cutoffWeight:
                    if verbose:
                        print("dummy inc stat", inc_stat)
                    db[ht][i][1].remove_cov(incstat.name)
                    db[ht][i][0]=None

                    remove_flag=True

                if inc_stat2.weight()<cutoffWeight:
                    if verbose:
                        print("dummy inc stat", inc_stat2)
                    db[ht][i][0].remove_cov(inc_stat2.name)
                    db[ht][i][1]=None

                    remove_flag=True

                if remove_flag:
                    db[ht+"_cov"][i]=None

    def updateGetStats(self, IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize):
        # Host BW: Stats on the srcIP's general Sender Statistics
        # Hstat = np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     Hstat[(i*3):((i+1)*3)] = self.HT_H.update_get_1D_Stats(srcIP, timestamp, datagramSize, self.Lambdas[i])

        #MAC.IP: Stats on src MAC-IP relationships
        if self.log_file is not None:
            self.log_file.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format( IPtype, srcMAC,dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, timestamp, datagramSize))

        self.prev_pkt_time=timestamp
        record=np.zeros(len(self.getNetStatHeaders()))

        for i in range(len(self.Lambdas)):
            record[i*3:(i+1)*3]=self.HT_MI.update_get_1D_Stats("{}_{}".format(srcMAC,srcIP), timestamp, datagramSize, i)
            record[i*7+15:(i+1)*7+15]=self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,i)
            record[(i*3)+50:((i+1)*3+50)]=self.HT_jit.update_get_1D_Stats("{}_{}".format(srcIP,dstIP), timestamp, 0,i,isTypeDiff=True)
            if srcProtocol == 'arp':
                record[(i*7)+65:((i+1)*7)+65]=self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, i)
            else:
                record[(i*7)+65:((i+1)*7)+65] = self.HT_Hp.update_get_1D2D_Stats("{}_{}".format(srcIP, srcProtocol), "{}_{}".format(dstIP , dstProtocol), timestamp, datagramSize, i)

        #
        # MIstat =  np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     MIstat[(i*3):((i+1)*3)] = self.HT_MI.update_get_1D_Stats("{}_{}".format(srcMAC,srcIP), timestamp, datagramSize, i)
        #
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HHstat =  np.zeros((7*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     HHstat[(i*7):((i+1)*7)] = self.HT_H.update_get_1D2D_Stats(srcIP, dstIP,timestamp,datagramSize,i)
        # # Host-Host Jitter:
        # HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
        # for i in range(len(self.Lambdas)):
        #     HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.update_get_1D_Stats("{}_{}".format(srcIP,dstIP), timestamp, 0,i,isTypeDiff=True)
        #
        # # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
        # HpHpstat =  np.zeros((7*len(self.Lambdas,)))
        # if srcProtocol == 'arp':
        #     for i in range(len(self.Lambdas)):
        #         HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats(srcMAC, dstMAC, timestamp, datagramSize, i)
        # else:  # some other protocol (e.g. TCP/UDP)
        #     for i in range(len(self.Lambdas)):
        #         HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.update_get_1D2D_Stats("{}_{}".format(srcIP, srcProtocol), "{}_{}".format(dstIP , dstProtocol), timestamp, datagramSize, i)

        self.num_updated+=1

        #clean our records every 100 updates
        if self.num_updated%self.clean_up_round==0:

            self.HT_MI.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_H.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_jit.cleanOutOldRecords(self.cutoffWeight, timestamp)
            self.HT_Hp.cleanOutOldRecords(self.cutoffWeight, timestamp)



        return record

    def get_stats(self, IPtype, srcMAC, dstMAC, srcIP, srcProtocol, dstIP, dstProtocol, t1, frame_len=None, Lambda=1):
        """get stats of a packet, framelen not needed"""

        MIstat =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            ID=srcMAC+srcIP
            MIstat[(i*3):((i+1)*3)] = self.HT_MI.get_1D_Stats(ID,self.Lambdas[i])

        HHstat =  np.zeros((7*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat[(i*7):((i+1)*7)] = self.HT_H.get_1D_Stats(srcIP,self.Lambdas[i])+self.HT_H.get_2D_stats(srcIP, dstIP,self.Lambdas[i],t1,level=2)


        # amount of traffic arriving out of time order
        HHstat_jit =  np.zeros((3*len(self.Lambdas,)))
        for i in range(len(self.Lambdas)):
            HHstat_jit[(i*3):((i+1)*3)] = self.HT_jit.get_1D_Stats(srcIP+dstIP,self.Lambdas[i])

        HpHpstat =  np.zeros((7*len(self.Lambdas,)))
        if srcProtocol == 'arp':
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.get_1D_Stats(srcMAC,self.Lambdas[i])+self.HT_Hp.get_2D_stats(srcMAC, dstMAC,self.Lambdas[i],t1,level=2)
        else:  # some other protocol (e.g. TCP/UDP)
            for i in range(len(self.Lambdas)):
                HpHpstat[(i*7):((i+1)*7)] = self.HT_Hp.get_1D_Stats(srcIP + srcProtocol, self.Lambdas[i])+self.HT_Hp.get_2D_stats(srcIP + srcProtocol, dstIP + dstProtocol,self.Lambdas[i],t1,level=2)

        return np.concatenate((MIstat,HHstat, HHstat_jit,HpHpstat))


    def getNetStatHeaders(self):

        MIstat_headers = self.HT_MI.get_headers()
        HHstat_headers = self.HT_H.get_headers(True)
        HHjitstat_headers = self.HT_jit.get_headers()
        HpHpstat_headers = self.HT_Hp.get_headers(True)


        return MIstat_headers + HHstat_headers + HHjitstat_headers + HpHpstat_headers
