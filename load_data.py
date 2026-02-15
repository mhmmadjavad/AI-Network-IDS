import pandas as pd
import os

def load_raw_train():
    """
    Infects raw KDD dataset from the local directory.
    Ensures column names are correctly assigned for downstream processing.
    """
    # Standard NSL-KDD feature names
    columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
              "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
              "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
              "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
              "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
              "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
              "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    path = 'DATA/raw/KDDTest+.txt'
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset at {path}")

    # Load data without header, assigning defined column names
    df = pd.read_csv(path, names=columns)
    return df