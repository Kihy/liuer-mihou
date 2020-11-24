from after_image.feature_extractor import *
from tqdm import tqdm
np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format})


def parse_kitsune(pcap_file, output_file_name, add_label=False, write_prob=1, count=float('Inf'), parse_type="scapy"):
    """Short summary.

    Args:
        pcap_file (string): path to pcap file..
        output_file_name (string): output path of the feature file. .
        add_label (boolean): whether to add label at the end of feature. if true, label is the filename. Defaults to False.
        write_prob (float 0~1): probability of writing a feature to file. 1 indicates all features are written. Defaults to 1.
        count (int): umber of packets to process. Defaults to float('Inf').
        parse_type (string): either scapy or tshark. There are some bugs with tshark parsing so stick with scapy. Defaults to "scapy".

    Returns:
        nothing, file is written to output_file_name

    Raises:
        EOFError,ValueError,StopIteration: when EOF is reached. already handled

    """

    feature_extractor = FE(pcap_file, parse_type=parse_type)
    # temp=open("tmp.txt","w")
    headers = feature_extractor.nstat.getNetStatHeaders()

    output_file = open(output_file_name, "w")
    label = output_file_name.split('/')[-1]

    if add_label:
        headers += ["label"]

    # print(headers)
    np.savetxt(output_file, [headers], fmt="%s", delimiter=",")

    skipped = 0
    written = 0
    t = tqdm(total=count)
    pkt_index = 0
    while pkt_index < count:
        try:
            if parse_type == "scapy":
                traffic_data, packet = feature_extractor.get_next_vector()
            else:
                traffic_data = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            print(e)
            break
        except ValueError as e:
            print("EOF Reached")
            print(e)
            break
        except StopIteration as e:
            print(e)
            print("EOF Reached")
            break

        pkt_index += 1
        t.update(1)
        if traffic_data == []:
            np.savetxt(output_file, [np.full(
                features.shape, -1)], delimiter=",")
            print(pkt_index)
            skipped += 1
            continue
        # print(traffic_data)
        features = feature_extractor.nstat.updateGetStats(*traffic_data)

        if np.isnan(features).any():
            print(features)
            break
        # temp.write("{}\n".format(pkt_index))
        if np.random.uniform(0, 1) < write_prob:
            if add_label:
                np.savetxt(output_file, [features], delimiter=",", newline=",")
                np.savetxt(output_file, [label], fmt="%s")
            else:
                np.savetxt(output_file, [features], delimiter=",")
        written += 1

    output_file.close()
    print("skipped:", skipped)
    print("written:", written)


if __name__ == '__main__':
    file_name = ["../ku_dataset/port_scan_attack_only", "../ku_dataset/flooding_attacker_only",
                 "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only", "../experiment/traffic_shaping/init_pcap/google_home_normal"]
    for i in file_name:
        print("processing file:", i)
        parse_kitsune(i + ".pcap", i + ".csv", False, parse_type="scapy")
