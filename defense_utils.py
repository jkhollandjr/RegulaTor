import pickle
import numpy as np

def get_trace(file_name, cutoff_time, cutoff_length):
    '''Given a filename, returns a list representation for processing'''

    all_lines = []
    with open(file_name) as fptr:
        for line in fptr:
            time = float(line.strip().split('\t')[0])
            direction = line.strip().split('\t')[1]

            if int(direction) > 0:
                direction = np.int(1)
            else:
                direction = np.int(-1)

            website = file_name.split('-')[0]
            trace_line = (time, direction, website, file_name)

            if time < cutoff_time:
                all_lines.append(trace_line)

    if len(all_lines) > cutoff_length:
        all_lines = all_lines[:cutoff_length]

    return all_lines

def get_download_packets(trace):
    '''Takes trace and returns list of timestamps of download packets'''
    output_trace = [packet[0] for packet in trace if packet[1] == -1]
    return output_trace

def get_upload_packets(trace):
    '''Takes trace and returns list of timestamps of download packets'''
    output_trace = [packet[0] for packet in trace if packet[1] == 1]
    return output_trace

def get_time_gaps(trace):
    '''Convert 1-d list of times into 1-d list of gaps'''
    output_trace = []
    current_time = 0.0
    for packet in trace:
        output_trace.append(float(packet) - current_time)
        current_time = float(packet)
    return output_trace

def output_pkl(trace_list, website_list, save_path):
    '''Output pkl file containing defended traces'''
    #create train and test splits
    first_split = int((len(trace_list) * 4 / 5))
    second_split = first_split + int(len(trace_list) / 10)

    trace_train = trace_list[:first_split]
    trace_valid = trace_list[first_split:second_split]
    trace_test = trace_list[second_split:]

    web_train = website_list[:first_split]
    web_valid = website_list[first_split:second_split]
    web_test = website_list[second_split:]
    print("write out to file")

    #write out to file
    trace_train_out = open(save_path + "X_train.pkl", "wb")
    pickle.dump(trace_train, trace_train_out)
    print("half train writtten")

    web_train_out = open(save_path + "y_train.pkl", "wb")
    pickle.dump(web_train, web_train_out)

    trace_valid_out = open(save_path + "X_valid.pkl", "wb")
    pickle.dump(trace_valid, trace_valid_out)

    web_valid_out = open(save_path + "y_valid.pkl", "wb")
    pickle.dump(web_valid, web_valid_out)

    trace_test_out = open(save_path + "X_test.pkl", "wb")
    pickle.dump(trace_test, trace_test_out)

    web_test_out = open(save_path + "y_test.pkl", "wb")
    pickle.dump(web_test, web_test_out)
    print("full train written...exiting")

