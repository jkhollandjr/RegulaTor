from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
import sys
import numpy as np
import argparse
from defense_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('source_path', help='Undefended dataset')
parser.add_argument('output_path', help='Output path')
parser.add_argument('--n_processes', type=int, help='Number of python processes to run in parallel', default='4')
parser.add_argument('--orig_rate', help='Original packet surge rate', default='277')
parser.add_argument('--dep_rate', help='Packet sending depreciation rate', default='.94')
parser.add_argument('--budget', type=int, help='Maximum possible padding budget', default='3550')
parser.add_argument('--threshold', help='Burst threshold', default='3.55')
parser.add_argument('--upload_ratio', help='Ratio of download packets to upload packets', default='3.95')
parser.add_argument('--delay_cap', help='Maximum upload packet delay', default='1.77')
args = parser.parse_args()

CUTOFF_LENGTH = 20000
CUTOFF_TIME = 120
SAVE_PICKLE = False

def regulator_download(target_trace):
    orig_rate = float(args.orig_rate)
    depreciation_rate = float(args.dep_rate)
    max_padding_budget = int(args.budget)
    burst_threshold = float(args.threshold)

    padding_budget = np.random.randint(0,max_padding_budget)

    output_trace = []
    upload_trace = []
 
    position = 10

    #send packets at a constant rate initially (to construct circuit)
    download_start = target_trace[position]
    added_packets = int(download_start*10)
    for i in range(added_packets):
        pkt_time = i*.1
        output_trace.append(pkt_time)

    output_trace.append(target_trace[position])
    current_time = download_start
    burst_time = target_trace[position]
    
    padding_packets = 0
    position = 1
    
    while True:
        #calculate target rate
        target_rate = orig_rate * (depreciation_rate**(current_time - burst_time))
        
        if(target_rate < 1):
            target_rate = 1        
        
        #if the original trace has been completely sent
        if(position == len(target_trace)):
            break
        
        #find number of real packets waiting to be sent
        queue_length = 0
        for c in range(position, len(target_trace)):
            if(target_trace[c] < current_time):
                queue_length += 1
            else:
                break

        #if waiting packets exceeds treshold, then begin a new burst
        if(queue_length > (burst_threshold*target_rate)):
            burst_time = current_time
        
        #calculate gap
        gap = 1 / float(target_rate)
        current_time += gap
        
        if(queue_length == 0 and padding_packets >= padding_budget):
            #no packets waiting and padding budget reached
            continue
        elif(queue_length == 0 and padding_packets < padding_budget):
            #no packets waiting, but padding budget not reached
            output_trace.append(current_time)
            padding_packets += 1
        else:
            #real packet to send
            output_trace.append(current_time)
            position += 1
       
    return output_trace

def regulator_upload_full(download_trace, upload_trace):

    upload_ratio = float(args.upload_ratio)
    delay_cap = float(args.delay_cap)

    output_trace = []

    #send one upload packet for every $upload_ratio download packets 
    upload_size = int(len(download_trace)/upload_ratio)
    output_trace = list(np.random.choice(download_trace, upload_size))

    #send at constant rate at first
    download_start = download_trace[10]
    added_packets = int(download_start*5)
    for i in range(added_packets):
        pkt_time = i*.2
        output_trace.append(pkt_time)

    #assign each packet to the next scheduled sending time in the output trace
    output_trace = sorted(output_trace)
    delay_packets = []
    packet_position = 0
    for t in upload_trace:
        found_packet = False
        for p in range(packet_position+1, len(output_trace)):
            if(output_trace[p] >= t and (output_trace[p]-t) < delay_cap):
                packet_position = p
                found_packet = True
                break
 
        #cap delay at delay_cap seconds
        if(found_packet == False):
            delay_packets.append(t+delay_cap)     
   
    output_trace += delay_packets
    
    return sorted(output_trace)

def cost_calc(orig_trace, alt_trace):
    '''calculate the bandwidth and latency overhead for download traces'''
    dummy_padding = len(alt_trace) - len(orig_trace)
    
    latency_cost = 0.0
    sending_time = 0.0
    last_packet_sent = 0
    last_packet_latency = 0.0
    for t in orig_trace:
        #find next available packet in sending schedule
        available = alt_trace[last_packet_sent:]
        for p in available:
            if(p > t or p == t):
                sending_time = p
                last_packet_sent = alt_trace.index(p)+1
                break
        
        latency_cost += (sending_time - t)
        #finds latency of last real packet
        last_packet_latency = (sending_time-t)
    
    return (dummy_padding, latency_cost, last_packet_latency)

def cost_calc_max_latency(orig_trace, alt_trace):
    '''calculates latency overhead for upload trace using a more pessimistic method'''
    dummy_padding = len(alt_trace) - len(orig_trace)
    
    latency_cost = 0.0
    sending_time = 0.0
    last_packet_sent = 0
    last_packet_latency = 0.0
    max_packet_latency = 0.0
    counter = 0
    location = 0
    for t in orig_trace:
        #find next available packet in sending schedule
        available = alt_trace[last_packet_sent:]
        for p in available:
            if(p > t or p == t):
                sending_time = p
                last_packet_sent = alt_trace.index(p)+1
                break
        
        latency_cost += (sending_time - t)
        
        #find packet with largest delay
        if((sending_time - t) > max_packet_latency and counter > 10):
            max_packet_latency = (sending_time - t)
            location = counter
        counter += 1
        last_packet_latency = (sending_time-t)
    
    return max_packet_latency, location

def simulate(file_name):
    trace = get_trace(args.source_path + str(file_name), CUTOFF_TIME, CUTOFF_LENGTH)
    website = int(file_name.split('-')[0])
    
    #get download and upload separately
    download_packets = get_download_packets(trace)
    upload_packets = get_upload_packets(trace)
    original_bandwidth = len(upload_packets) + len(download_packets)

    #get defended traces
    padded_download = regulator_download(download_packets)
    padded_upload = regulator_upload_full(padded_download, upload_packets)
    padded_bandwidth = len(padded_download) + len(padded_upload)

    #calculate latency overhead
    _, _, download_latency_overhead = cost_calc(download_packets, padded_download) 
    upload_latency_overhead, _ = cost_calc_max_latency(upload_packets, padded_upload)
    latency_overhead = download_latency_overhead + upload_latency_overhead

    download_packets = [(p, -1) for p in padded_download]
    upload_packets = [(p, 1) for p in padded_upload]

    both_output = sorted(download_packets + upload_packets, key=lambda x: x[0])

    #output to file
    path = args.output_path + str(file_name)
    with open(path, 'w') as w:
        for p in both_output:
            w.write(str(p[0]) + '\t' + str(p[1]) + '\n')

    if(SAVE_PICKLE):
        #output .pkl files
        direction_only = [np.float64(x[1]) for x in both_output]

        #pad to 5000 (for deep fingerprinting attack)
        if(len(direction_only) > 5000):
            direction_only = direction_only[:5000]
        else:
            direction_only += [0]*(5000-len(direction_only))

        return np.asarray(direction_only), np.int64(website)

if __name__ == '__main__':    
    file_list = [f for f in listdir(args.source_path) if isfile(join(args.source_path, f))]
    
    p = Pool(args.n_processes)
    
    if(SAVE_PICKLE):
        all_indiv_streams = []
        all_indiv_streams = p.map(simulate, file_list)
        all_indiv_streams_real = []
        for x in all_indiv_streams:
            if x is not None:
                all_indiv_streams_real.append(x)
        all_indiv_streams = all_indiv_streams_real

        website_list = [x[1] for x in all_indiv_streams]
        trace_list = [x[0] for x in all_indiv_streams] 
        
        output_pkl(trace_list, website_list, args.output_path) 

    else:
        p.map(simulate, file_list)
