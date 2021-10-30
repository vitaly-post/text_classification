from cred_config import scp as default_conf
from paramiko import SSHClient
import paramiko
from scp import SCPClient
import sys

class ScpHandler:
    def __init__(self, conf=default_conf):
        self.host = conf['host']
        self.username = conf['username']
        self.password = conf['password']
        self.remote_path = conf['remote_path']
        self.remote_model_name = conf['remote_model_name']
        self.local_model_path = conf['local_model_path']

        self.have_backup = False

        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(self.host, username=self.username, password=self.password)
        # SCPCLient takes a paramiko transport as an argument
        self.scp = SCPClient(ssh.get_transport(), progress=self.progress)

    def send_model_to_remote(self):
        if self.have_backup:
            self.scp.put(f'{self.local_model_path}{self.remote_model_name}', f'{self.remote_path}{self.remote_model_name}')
        else:
            print('Need to make a backup!')

    # Define progress callback that prints the current percentage completed for the file
    def progress(self, filename, size, sent):
        sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100))

    def get_model_from_remote(self):
        from time import gmtime, strftime

        dt = strftime("%d%m%Y_%H%M%S", gmtime())
        self.scp.get(f'{self.remote_path}{self.remote_model_name}', f'./backup/{self.remote_model_name}_{dt}')
        self.scp.close()
        self.have_backup = True