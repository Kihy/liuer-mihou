from after_image.feature_extractor import *
from tqdm import tqdm
import unittest
import numpy as np



class TestStringMethods(unittest.TestCase):

    @unittest.skip("too time consuming, dont run often")
    def test_get_stats(self):
        pcap_path="../../TrafficManipulator/example/test.pcap"
        feature_extractor=FE(pcap_path)

        for i in tqdm(range(feature_extractor.num_pkt)):
            features = feature_extractor.get_next_vector()

            true_stats=feature_extractor.nstat.updateGetStats(*features)

            test_stats=feature_extractor.nstat.get_stats(*features)

            np.testing.assert_almost_equal(true_stats, test_stats)

    def test_roll_back(self):
        pcap_path="../../TrafficManipulator/example/test.pcap"
        feature_extractor=FE(pcap_path, parse_type="scapy")

        #update both nstat with features

        features, packet = feature_extractor.get_next_vector()
        while features == []:
            features, packet = feature_extractor.get_next_vector()

        feature1 =feature_extractor.nstat.updateGetStats(*features)

        # check if behaviour is the same
        db=feature_extractor.nstat.get_records(*features)
        dummy_db=copy.deepcopy(db)


        f1=feature_extractor.nstat.update_dummy_db(0, '3c:33:00:98:ee:fd', '40:8d:5c:4b:99:14', '192.168.2.110', '23', '192.168.2.107', '57206', 1540450874.471256, 70,dummy_db)

        f2=feature_extractor.nstat.updateGetStats(0, '3c:33:00:98:ee:fd', '40:8d:5c:4b:99:14', '192.168.2.110', '23', '192.168.2.107', '57206', 1540450874.471256, 70 )

        assert (f1==f2).all()

        db=feature_extractor.nstat.get_records(*features)
        dummy_db=copy.deepcopy(db)


        # simulate adding adversarial crafted packets
        feature_extractor.nstat.update_dummy_db(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', '23', '192.168.2.152', '57206', 1540450873.46, 70 ,dummy_db)
        feature_extractor.nstat.update_dummy_db(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', '23', '192.168.2.152', '57206', 1540450873.46, 70 ,dummy_db)
        feature=feature_extractor.nstat.update_dummy_db(0, '3c:33:00:98:ee:fd', 'ff:ff:ff:ff:ff:ff', '192.168.2.110', '23', '192.168.2.152', '57206', 1540450873.46, 70 ,dummy_db)

        # nstat and dummy_nstat should not be the same
        assert (feature1 != feature).any()

        db2=feature_extractor.nstat.get_records(*features)
        # should be the same
        assert db == db2






if __name__ == '__main__':
    unittest.main()
