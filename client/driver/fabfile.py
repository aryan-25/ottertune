#
# OtterTune - fabfile.py
#
# Copyright (c) 2017-18, Carnegie Mellon University Database Group
#
'''
Created on Mar 23, 2018

@author: bohan
'''
import glob
import json
import logging
import os
import re
import sys
import time
import warnings
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from multiprocessing import Process

import requests
from cryptography.utils import CryptographyDeprecationWarning
from scp import SCPClient

with warnings.catch_warnings():
    warnings.simplefilter("ignore", CryptographyDeprecationWarning)
    import paramiko
    from fabric.api import env, hide, lcd, local, settings, show, task
    from fabric.state import output as fabric_output
    # from utils import (FabricException, file_exists, get, get_content,
    #                    load_driver_conf, parse_bool, put, run, run_sql_script,
    #                    sudo)
    from utils import (FabricException, load_driver_conf, parse_bool, get_content)

# Loads the driver config file (defaults to driver_config.py)
dconf = load_driver_conf()  # pylint: disable=invalid-name

# Fabric settings
fabric_output.update({
    'running': True,
    'stdout': True,
})
env.abort_exception = FabricException
env.hosts = [dconf.LOGIN]
env.password = dconf.LOGIN_PASSWORD

# Create local directories
for _d in (dconf.RESULT_DIR, dconf.LOG_DIR, dconf.TEMP_DIR):
    os.makedirs(_d, exist_ok=True)


# Configure logging
LOG = logging.getLogger(__name__)
LOG.setLevel(getattr(logging, dconf.LOG_LEVEL, logging.DEBUG))
Formatter = logging.Formatter(  # pylint: disable=invalid-name
    fmt='%(asctime)s [%(funcName)s:%(lineno)03d] %(levelname)-5s: %(message)s',
    datefmt='%m-%d-%Y %H:%M:%S')
ConsoleHandler = logging.StreamHandler()  # pylint: disable=invalid-name
ConsoleHandler.setFormatter(Formatter)
LOG.addHandler(ConsoleHandler)
FileHandler = RotatingFileHandler(  # pylint: disable=invalid-name
    dconf.DRIVER_LOG, maxBytes=50000, backupCount=2)
FileHandler.setFormatter(Formatter)
LOG.addHandler(FileHandler)


@task
def check_disk_usage(db_config: dict):
    partition = dconf.DATABASE_DISK
    disk_use = 0
    if partition:
        cmd = "df -h {}".format(partition)
        out = updated_sudo(cmd, hostname=db_config['host'], username=db_config['username'], password=db_config['password']).splitlines()[1]
        m = re.search(r'\d+(?=%)', out)
        if m:
            disk_use = int(m.group(0))
        LOG.info("Current Disk Usage: %s%s", disk_use, '%')
    return disk_use


@task
def check_memory_usage():
    updated_sudo('free -m -h')


# @task
def create_controller_config(db_config: dict):
    if dconf.DB_TYPE == 'postgres':
        dburl_fmt = 'jdbc:postgresql://{host}:{port}/{db}'.format
    elif dconf.DB_TYPE == 'oracle':
        dburl_fmt = 'jdbc:oracle:thin:@{host}:{port}:{db}'.format
    elif dconf.DB_TYPE == 'mysql':
        if dconf.DB_VERSION in ['5.6', '5.7']:
            dburl_fmt = 'jdbc:mysql://{host}:{port}/{db}?useSSL=false'.format
        elif dconf.DB_VERSION == '8.0':
            dburl_fmt = ('jdbc:mysql://{host}:{port}/{db}?'
                        'allowPublicKeyRetrieval=true&useSSL=false').format
        else:
            raise Exception("MySQL Database Version {} Not Implemented !".format(dconf.DB_VERSION))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))

    config = dict(
        database_type=dconf.DB_TYPE,
        database_url=dburl_fmt(host=db_config['host'], port=dconf.DB_PORT, db=dconf.DB_NAME),
        username=dconf.DB_USER,
        password=dconf.DB_PASSWORD,
        upload_code='DEPRECATED',
        upload_url='DEPRECATED',
        workload_name=dconf.OLTPBENCH_BENCH
    )

    with open(db_config['controller_config_file'], 'w') as f:
        json.dump(config, f, indent=2)


@task
def restart_database(db_config: dict):
    if dconf.DB_TYPE == 'postgres':
        if db_config['connection_type'] == 'docker':
            # Restarting the docker container here is the cleanest way to do it
            # becaues there's no init system running and the only process running
            # in the container is postgres itself
            local('docker restart {}'.format(dconf.CONTAINER_NAME))
        elif db_config['connection_type'] == 'remote_docker':
            updated_sudo('docker restart {}'.format(dconf.CONTAINER_NAME), hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
        else:
            updated_sudo('sudo systemctl restart postgresql@9.6-main', hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
            # sudo('systemctl restart postgresql@16-main', user=dconf.ADMIN_USER, capture=False)
    elif dconf.DB_TYPE == 'mysql':
        if db_config['connection_type'] == 'docker':
            local('docker restart {}'.format(dconf.CONTAINER_NAME))
        elif db_config['connection_type'] == 'remote_docker':
            updated_sudo('docker restart {}'.format(dconf.CONTAINER_NAME), hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
        else:
            updated_sudo('service mysql restart', hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
    elif dconf.DB_TYPE == 'oracle':
        db_log_path = os.path.join(os.path.split(dconf.DB_CONF)[0], 'startup.log')
        local_log_path = os.path.join(dconf.LOG_DIR, 'startup.log')
        local_logs_path = os.path.join(dconf.LOG_DIR, 'startups.log')
        run_sql_script('restartOracle.sh', db_log_path)
        get(db_log_path, local_log_path)
        with open(local_log_path, 'r') as fin, open(local_logs_path, 'a') as fout:
            lines = fin.readlines()
            for line in lines:
                if line.startswith('ORACLE instance started.'):
                    return True
                if not line.startswith('SQL>'):
                    fout.write(line)
            fout.write('\n')
        return False
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    return True


@task
def drop_database(db_config: dict):
    if dconf.DB_TYPE == 'postgres':
        updated_sudo(
            "PGPASSWORD={} dropdb -e --if-exists {} -U {} -h {}".format(
                dconf.DB_PASSWORD, dconf.DB_NAME, dconf.DB_USER, dconf.DB_HOST
            ),
            hostname=db_config['host'],
            username=db_config['username'],
            password=db_config['password']
        )
    elif dconf.DB_TYPE == 'mysql':
        run(
            "mysql --user={} --password={} -e 'drop database if exists {}'".format(
            dconf.DB_USER, dconf.DB_PASSWORD, dconf.DB_NAME)
        )
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))


@task
def create_database(db_config: dict):
    if dconf.DB_TYPE == 'postgres':
        updated_sudo(
            "PGPASSWORD={} createdb -e {} -U {} -h {}".format(
                dconf.DB_PASSWORD, dconf.DB_NAME, dconf.DB_USER, dconf.DB_HOST
            ),
            hostname=db_config['host'],
            username=db_config['username'],
            password=db_config['password']
        )
    elif dconf.DB_TYPE == 'mysql':
        run("mysql --user={} --password={} -e 'create database {}'".format(
            dconf.DB_USER, dconf.DB_PASSWORD, dconf.DB_NAME))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))


@task
def create_user(db_config: dict):
    if dconf.DB_TYPE == 'postgres':
        sql = "CREATE USER {} SUPERUSER PASSWORD '{}';".format(dconf.DB_USER, dconf.DB_PASSWORD)
        updated_sudo(
            "PGPASSWORD={} psql -c \\\"{}\\\" -U postgres -h {}".format(
                dconf.DB_PASSWORD, sql, dconf.DB_HOST
            ),
            hostname=db_config['host'],
            username=db_config['username'],
            password=db_config['password']
        )
    elif dconf.DB_TYPE == 'oracle':
        run_sql_script('createUser.sh', dconf.DB_USER, dconf.DB_PASSWORD)
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))


@task
def drop_user(db_config: dict):
    if dconf.DB_TYPE == 'postgres':
        sql = "DROP USER IF EXISTS {};".format(dconf.DB_USER)
        updated_sudo(
            "PGPASSWORD={} psql -c \\\"{}\\\" -U postgres -h {}".format(
                dconf.DB_PASSWORD, sql, dconf.DB_HOST
            ),
            hostname=db_config['host'],
            username=db_config['username'],
            password=db_config['password']
        )
    elif dconf.DB_TYPE == 'oracle':
        run_sql_script('dropUser.sh', dconf.DB_USER)
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))

@task
def reset_conf(db_config: dict, always=True):
    if always:
        change_conf(db_config=db_config)
        return

    # reset the config only if it has not been changed by Ottertune,
    # i.e. OtterTune signal line is not in the config file.
    signal = "# configurations recommended by ottertune:\n"
    tmp_conf_in = os.path.join(dconf.TEMP_DIR, f"{db_config['id']}_" + os.path.basename(dconf.DB_CONF) + '.in')
    updated_get(dconf.DB_CONF, tmp_conf_in, hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
    with open(tmp_conf_in, 'r') as f:
        lines = f.readlines()
    if signal not in lines:
        change_conf(db_config=db_config)


def updated_get(remote_location: str, local_location: str, hostname: str, username: str, password: str):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, port=22, username=username, password=password)
    print(f'Fetching from {hostname}:{remote_location} to {local_location}')

    with SCPClient(ssh_client.get_transport()) as scp:
        scp.get(remote_location, local_location)
        print(f'File downloaded successfully from {hostname}:{remote_location} to {local_location}')


def updated_put(local_location: str, remote_location: str, hostname: str, username: str, password: str):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, port=22, username=username, password=password)

    with SCPClient(ssh_client.get_transport()) as scp:
        scp.put(local_location, remote_location)
        print(f'File uploaded successfully from {local_location} to {remote_location}')


def updated_sudo(cmd: str, hostname: str, username: str, password: str):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, port=22, username=username, password=password)

    _, stdout, stderr = ssh_client.exec_command(cmd)

    for line in stdout:
        print(line.strip('\n'))

    for err in stderr:
        print(err.strip('\n'), file=sys.stderr)

    ssh_client.close()

def updated_file_exists(filename: str, hostname: str, username: str, password: str):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=hostname, port=22, username=username, password=password)

    sftp = ssh_client.open_sftp()
    try:
        sftp.stat(filename)
    except FileNotFoundError as e:
        print(f'File {filename} does not exist on {hostname}: {filename}, {e}')
        return False
    return True

@task
def change_conf(db_config: dict, next_conf=None):
    signal = "# configurations recommended by ottertune:\n"
    next_conf = next_conf or {}

    tmp_conf_in = os.path.join(dconf.TEMP_DIR, f"{db_config['id']}_" + os.path.basename(dconf.DB_CONF) + '.in')
    updated_get(dconf.DB_CONF, tmp_conf_in, hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
    with open(tmp_conf_in, 'r') as f:
        lines = f.readlines()

    if signal not in lines:
        lines += ['\n', signal]

    signal_idx = lines.index(signal)
    lines = lines[0:signal_idx + 1]

    if dconf.DB_TYPE == 'mysql':
        lines.append('[mysqld]\n')

    if dconf.BASE_DB_CONF:
        assert isinstance(dconf.BASE_DB_CONF, dict), \
            (type(dconf.BASE_DB_CONF), dconf.BASE_DB_CONF)
        for name, value in sorted(dconf.BASE_DB_CONF.items()):
            if value is None:
                lines.append('{}\n'.format(name))
            else:
                lines.append('{} = {}\n'.format(name, value))

    if isinstance(next_conf, str):
        with open(next_conf, 'r') as f:
            recommendation = json.load(
                f, encoding="UTF-8", object_pairs_hook=OrderedDict)['recommendation']
    else:
        recommendation = next_conf

    assert isinstance(recommendation, dict)

    for name, value in recommendation.items():
        if dconf.DB_TYPE == 'oracle' and isinstance(value, str):
            value = value.strip('B')
        # If innodb_flush_method is set to NULL on a Unix-like system,
        # the fsync option is used by default.
        if name == 'innodb_flush_method' and value == '':
            value = "fsync"
        lines.append('{} = {}\n'.format(name, value))
    lines.append('\n')

    tmp_conf_out = os.path.join(dconf.TEMP_DIR, f"{db_config['id']}_" + os.path.basename(dconf.DB_CONF) + '.out')
    with open(tmp_conf_out, 'w') as f:
        f.write(''.join(lines))

    updated_sudo('cp {0} {0}.ottertune.bak'.format(dconf.DB_CONF), hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
    updated_put(tmp_conf_out, dconf.DB_CONF, hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
    local('rm -f {} {}'.format(tmp_conf_in, tmp_conf_out))


@task
def load_oltpbench(db_config: dict):
    if os.path.exists(db_config['oltpbench_config_file']) is False:
        msg = 'oltpbench config {} does not exist, '.format(db_config['oltpbench_config_file'])
        msg += 'please double check the option in driver_config.py'
        raise Exception(msg)
    cmd = f"cd ../../benchbase/target/benchbase-postgres/ && java -jar benchbase.jar -b tpcc -c {db_config['oltpbench_config_file']} --create=true --load=true"
    with lcd(dconf.OLTPBENCH_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_oltpbench(db_config: dict):
    if os.path.exists(db_config['oltpbench_config_file']) is False:
        msg = 'oltpbench config {} does not exist, '.format(db_config['oltpbench_config_file'])
        msg += 'please double check the option in driver_config.py'
        raise Exception(msg)
    cmd = f"cd ../../benchbase/target/benchbase-postgres/ && java -jar benchbase.jar -b tpcc -c {db_config['oltpbench_config_file']} --execute=true -s 5"
    with lcd(dconf.OLTPBENCH_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def run_oltpbench_bg(db_config: dict):
    if os.path.exists(db_config['oltpbench_config_file']) is False:
        msg = 'oltpbench config {} does not exist, '.format(db_config['oltpbench_config_file'])
        msg += 'please double check the option in driver_config.py'
        raise Exception(msg)
    # set oltpbench config, including db username, password, url
    cmd = "cd ../../benchbase/target/benchbase-postgres/ && java -jar benchbase.jar -b tpcc -c {} --execute=true -s 5 > {} 2>&1 &".format(
        db_config['oltpbench_config_file'], db_config['oltpbench_log_file']
    )
    with lcd(dconf.OLTPBENCH_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def signal_controller(db_config: dict):
    pidfile = os.path.join(dconf.CONTROLLER_HOME, f"{db_config['id']}_pid.txt")
    with open(pidfile, 'r') as f:
        pid = int(f.read())
    cmd = 'kill -2 {}'.format(pid)
    with lcd(dconf.CONTROLLER_HOME):  # pylint: disable=not-context-manager
        local(cmd)


@task
def save_dbms_result(db_config: dict):
    t = int(time.time())
    files = ['knobs.json', 'metrics_after.json', 'metrics_before.json', 'summary.json']
    if dconf.ENABLE_UDM:
        files.append('user_defined_metrics.json')
    for f_ in files:
        srcfile = os.path.join(dconf.CONTROLLER_HOME, f"{db_config['id']}_output", f_)
        dstfile = os.path.join(dconf.RESULT_DIR, '{}__{}__{}'.format(db_config['id'], t, f_))
        local('cp {} {}'.format(srcfile, dstfile))
    return t


@task
def save_next_config(next_config, db_config: dict, t=None):
    if not t:
        t = int(time.time())
    with open(os.path.join(dconf.RESULT_DIR, '{}__{}__next_config.json'.format(db_config['id'], t)), 'w') as f:
        json.dump(next_config, f, indent=2)
    return t


@task
def free_cache(db_config: dict):
    if db_config['connection_type'] not in ['docker', 'remote_docker']:
        with show('everything'), settings(warn_only=True):  # pylint: disable=not-context-manager
            res = updated_sudo("sh -c \"echo 3 > /proc/sys/vm/drop_caches\"", hostname=db_config['host'], username=db_config['username'], password=db_config['password'])
            # if res.failed:
            # LOG.error('%s (return code %s)', res.stderr.strip(), res.return_code)
    else:
        res = updated_sudo("sh -c \"echo 3 > /proc/sys/vm/drop_caches\"", hostname=db_config['host'], username=db_config['username'], password=db_config['password'])


@task
def upload_result(result_dir=None, prefix=None, upload_code=None):
    result_dir = result_dir or os.path.join(dconf.CONTROLLER_HOME, f"{prefix}_output")
    prefix = prefix or ''
    upload_code = upload_code or dconf.UPLOAD_CODE
    files = {}
    bases = ['summary', 'knobs', 'metrics_before', 'metrics_after']
    if dconf.ENABLE_UDM:
        bases.append('user_defined_metrics')
    for base in bases:
        fpath = os.path.join(result_dir, base + '.json')

        # Replaces the true db version with the specified version to allow for
        # testing versions not officially supported by OtterTune
        if base == 'summary' and dconf.OVERRIDE_DB_VERSION:
            with open(fpath, 'r') as f:
                summary = json.load(f)
            summary['real_database_version'] = summary['database_version']
            summary['database_version'] = dconf.OVERRIDE_DB_VERSION
            with open(fpath, 'w') as f:
                json.dump(summary, f, indent=1)

        files[base] = open(fpath, 'rb')

    response = requests.post(dconf.WEBSITE_URL + '/new_result/', files=files,
                             data={'upload_code': upload_code})
    if response.status_code != 200:
        raise Exception('Error uploading result.\nStatus: {}\nMessage: {}\n'.format(
            response.status_code, get_content(response)))

    for f in files.values():  # pylint: disable=not-an-iterable
        f.close()

    content = get_content(response)

    LOG.info(content)

    return content


@task
def get_result(result_id, max_time_sec=180, interval_sec=5, upload_code=None):
    max_time_sec = int(max_time_sec)
    interval_sec = int(interval_sec)
    upload_code = upload_code or dconf.UPLOAD_CODE
    url = dconf.WEBSITE_URL + '/query_and_get/' + upload_code + '/' + str(result_id)
    elapsed = 0
    response_dict = None
    rout = ''

    print(f"Fetching from {url}")

    while elapsed <= max_time_sec:
        rsp = requests.get(url)
        response = get_content(rsp)
        assert response != 'null'
        rout = json.dumps(response, indent=4) if isinstance(response, dict) else response

        # LOG.debug('%s\n\n[status code: %d, type(response): %s, elapsed: %ds, %s]', rout,
        #           rsp.status_code, type(response), elapsed,
        #           ', '.join(['{}: {}'.format(k, v) for k, v in rsp.headers.items()]))

        if rsp.status_code == 200:
            # Success
            response_dict = response
            break

        elif rsp.status_code == 202:
            # Not ready
            time.sleep(interval_sec)
            elapsed += interval_sec

        elif rsp.status_code == 400:
            # Failure
            raise Exception(
                "Failed to download the next config.\nStatus code: {}\nMessage: {}\n".format(
                    rsp.status_code, rout))

        elif rsp.status_code == 500:
            # Failure
            msg = rout
            if isinstance(response, str):
                savepath = os.path.join(dconf.LOG_DIR, 'error.html')
                with open(savepath, 'w') as f:
                    f.write(response)
                msg = "Saved HTML error to '{}'.".format(os.path.relpath(savepath))
            raise Exception(
                "Failed to download the next config.\nStatus code: {}\nMessage: {}\n".format(
                    rsp.status_code, msg))

        else:
            raise NotImplementedError(
                "Unhandled status code: '{}'.\nMessage: {}".format(rsp.status_code, rout))

    if not response_dict:
        assert elapsed > max_time_sec, \
            'response={} but elapsed={}s <= max_time={}s'.format(
                rout, elapsed, max_time_sec)
        raise Exception(
            'Failed to download the next config in {}s: {} (elapsed: {}s)'.format(
                max_time_sec, rout, elapsed))

    LOG.info('Downloaded the next config in %ds', elapsed)

    return response_dict


@task
def download_debug_info(pprint=False):
    pprint = parse_bool(pprint)
    url = '{}/dump/{}'.format(dconf.WEBSITE_URL, dconf.UPLOAD_CODE)
    params = {'pp': int(True)} if pprint else {}
    rsp = requests.get(url, params=params)

    if rsp.status_code != 200:
        raise Exception('Error downloading debug info.')

    filename = rsp.headers.get('Content-Disposition').split('=')[-1]
    file_len, exp_len = len(rsp.content), int(rsp.headers.get('Content-Length'))
    assert file_len == exp_len, 'File {}: content length != expected length: {} != {}'.format(
        filename, file_len, exp_len)

    with open(filename, 'wb') as f:
        f.write(rsp.content)
    LOG.info('Downloaded debug info to %s', filename)

    return filename


@task
def add_udm(result_dir=None):
    result_dir = result_dir or os.path.join(dconf.CONTROLLER_HOME, 'output')
    with lcd(dconf.UDM_DIR):  # pylint: disable=not-context-manager
        local('python3 user_defined_metrics.py {}'.format(result_dir))


@task
def upload_batch(result_dir=None, sort=True, upload_code=None):
    result_dir = result_dir or dconf.RESULT_DIR
    sort = parse_bool(sort)
    results = glob.glob(os.path.join(result_dir, '*__summary.json'))
    if sort:
        results = sorted(results)
    count = len(results)

    LOG.info('Uploading %d samples from %s...', count, result_dir)
    for i, result in enumerate(results):
        prefix = os.path.basename(result)
        prefix_len = os.path.basename(result).find('_') + 2
        prefix = prefix[:prefix_len]
        upload_result(result_dir=result_dir, prefix=prefix, upload_code=upload_code)
        LOG.info('Uploaded result %d/%d: %s__*.json', i + 1, count, prefix)

@task
def dump_database(db_config: dict):
    dumpfile = os.path.join(dconf.DB_DUMP_DIR, dconf.DB_NAME + '.dump')
    if dconf.DB_TYPE == 'oracle':
        if not dconf.ORACLE_FLASH_BACK and updated_file_exists(dumpfile, db_config['host'], db_config['username'], db_config['password']):
            LOG.info('%s already exists ! ', dumpfile)
            return False
    else:
        if updated_file_exists(dumpfile, db_config['host'], db_config['username'], db_config['password']):
            LOG.info('%s already exists ! ', dumpfile)
            return False

    if dconf.DB_TYPE == 'oracle' and dconf.ORACLE_FLASH_BACK:
        LOG.info('create restore point %s for database %s in %s', dconf.RESTORE_POINT,
                 dconf.DB_NAME, dconf.RECOVERY_FILE_DEST)
    else:
        LOG.info('Dump database %s to %s', dconf.DB_NAME, dumpfile)

    if dconf.DB_TYPE == 'oracle':
        if dconf.ORACLE_FLASH_BACK:
            run_sql_script('createRestore.sh', dconf.RESTORE_POINT,
                           dconf.RECOVERY_FILE_DEST_SIZE, dconf.RECOVERY_FILE_DEST)
        else:
            run_sql_script('dumpOracle.sh', dconf.DB_USER, dconf.DB_PASSWORD,
                           dconf.DB_NAME, dconf.DB_DUMP_DIR)

    elif dconf.DB_TYPE == 'postgres':
        updated_sudo(
            'PGPASSWORD={} pg_dump --verbose -U {} -h {} -F c -d {} > {}'.format(
                dconf.DB_PASSWORD, dconf.DB_USER, dconf.DB_HOST, dconf.DB_NAME, dumpfile
            ),
            hostname=db_config['host'],
            username=db_config['username'],
            password=db_config['password']
        )
    elif dconf.DB_TYPE == 'mysql':
        sudo('mysqldump --user={} --password={} --databases {} > {}'.format(
            dconf.DB_USER, dconf.DB_PASSWORD, dconf.DB_NAME, dumpfile))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    return True


@task
def clean_recovery():
    run_sql_script('removeRestore.sh', dconf.RESTORE_POINT)
    cmds = ("""rman TARGET / <<EOF\nDELETE ARCHIVELOG ALL;\nexit\nEOF""")
    updated_sudo(cmds)


@task
def restore_database(db_config: dict):
    dumpfile = os.path.join(dconf.DB_DUMP_DIR, dconf.DB_NAME + '.dump')
    if not dconf.ORACLE_FLASH_BACK and not updated_file_exists(dumpfile, db_config['host'], db_config['username'], db_config['password']):
        raise FileNotFoundError("Database dumpfile '{}' does not exist!".format(dumpfile))

    LOG.info('Start restoring database')
    if dconf.DB_TYPE == 'oracle':
        if dconf.ORACLE_FLASH_BACK:
            run_sql_script('flashBack.sh', dconf.RESTORE_POINT)
            clean_recovery()
        else:
            drop_user()
            create_user()
            run_sql_script('restoreOracle.sh', dconf.DB_USER, dconf.DB_NAME)
    elif dconf.DB_TYPE == 'postgres':
        drop_database(db_config)
        create_database(db_config)
        updated_sudo(
            'PGPASSWORD={} pg_restore --verbose -j 4 -U {} -h {} -n public -F c -d {} {}'.format(
                dconf.DB_PASSWORD, dconf.DB_USER, dconf.DB_HOST, dconf.DB_NAME, dumpfile
            ),
            hostname=db_config['host'],
            username=db_config['username'],
            password=db_config['password']
        )
    elif dconf.DB_TYPE == 'mysql':
        run('mysql --user={} --password={} < {}'.format(dconf.DB_USER, dconf.DB_PASSWORD, dumpfile))
    else:
        raise Exception("Database Type {} Not Implemented !".format(dconf.DB_TYPE))
    LOG.info('Finish restoring database')


@task
def is_ready_db(interval_sec=10):
    if dconf.DB_TYPE == 'mysql':
        cmd_fmt = "mysql --user={} --password={} -e 'exit'".format
    else:
        LOG.info('database %s connecting function is not implemented, sleep %s seconds and return',
                 dconf.DB_TYPE, dconf.RESTART_SLEEP_SEC)
        return

    with hide('everything'), settings(warn_only=True):  # pylint: disable=not-context-manager
        while True:
            res = run(cmd_fmt(dconf.DB_USER, dconf.DB_PASSWORD))
            if res.failed:
                LOG.info('Database %s is not ready, wait for %s seconds',
                         dconf.DB_TYPE, interval_sec)
                time.sleep(interval_sec)
            else:
                LOG.info('Database %s is ready.', dconf.DB_TYPE)
                return

def _ready_to_start_oltpbench(db_config: dict):
    ready = False
    if os.path.exists(db_config['controller_log_file']):
        with open(db_config['controller_log_file'], 'r') as f:
            content = f.read()
        ready = 'Output the process pid to' in content
    return ready


def _ready_to_start_controller(db_config: dict):
    ready = False
    if os.path.exists(db_config['oltpbench_log_file']):
        with open(db_config['oltpbench_log_file'], 'r') as f:
            content = f.read()
        ready = 'Warmup complete, starting measurements' in content
    return ready


def _ready_to_shut_down_controller(db_config: dict):
    pidfile = os.path.join(dconf.CONTROLLER_HOME, f"{db_config['id']}_pid.txt")
    ready = False
    if os.path.exists(pidfile) and os.path.exists(db_config['oltpbench_log_file']):
        with open(db_config['oltpbench_log_file'], 'r') as f:
            content = f.read()
        if 'Failed' in content:
            m = re.search('\n.*Failed.*\n', content)
            error_msg = m.group(0)
            LOG.error('OLTPBench Failed!')
            return True, error_msg
        ready = 'Output samples into file' in content
    return ready, None


def clean_logs(db_config: dict):
    # remove oltpbench and controller log files
    local('rm -f {} {}'.format(db_config['oltpbench_log_file'], db_config['controller_log_file']))


@task
def clean_oltpbench_results():
    # remove oltpbench result files
    local('rm -f {}/results/outputfile*'.format(dconf.OLTPBENCH_HOME))


@task
def clean_controller_results():
    # remove oltpbench result files
    local('rm -f {}/output/*.json'.format(dconf.CONTROLLER_HOME))


@task
def loop(db_config: dict, i: int):
    i = int(i)

    # free cache
    free_cache(db_config)

    # remove oltpbench log and controller log
    clean_logs(db_config)

    if dconf.ENABLE_UDM is True:
        clean_oltpbench_results()

    # check disk usage
    if check_disk_usage(db_config) > dconf.MAX_DISK_USAGE:
        LOG.warning('Exceeds max disk usage %s', dconf.MAX_DISK_USAGE)

    create_controller_config(db_config)
    cmd = f"cd ../controller && gradle run -PappArgs='-c ./config/{db_config['id']}_postgres_config.json -t -1 -d ./{db_config['id']}_output/ -i {db_config['id']}' --no-daemon > {db_config['controller_log_file']}"
    # run controller from another process
    p = Process(target=local, args=(cmd,))
    p.start()
    LOG.info('Run the controller')

    # run oltpbench as a background job
    while not _ready_to_start_oltpbench(db_config):
        time.sleep(1)
    run_oltpbench_bg(db_config)
    LOG.info('Run OLTP-Bench')

    # the controller starts the first collection
    while not _ready_to_start_controller(db_config):
        time.sleep(1)
    signal_controller(db_config)
    LOG.info('Start the first collection')

    # stop the experiment
    ready_to_shut_down = False
    error_msg = None
    while not ready_to_shut_down:
        ready_to_shut_down, error_msg = _ready_to_shut_down_controller(db_config)
        time.sleep(1)

    signal_controller(db_config)
    LOG.info('Start the second collection, shut down the controller')

    p.join()
    if error_msg:
        raise Exception('OLTPBench Failed: ' + error_msg)
    # add user defined metrics
    if dconf.ENABLE_UDM is True:
        add_udm()

    # save result
    result_timestamp = save_dbms_result(db_config)

    if i >= dconf.WARMUP_ITERATIONS:
        # upload result
        upload_response = upload_result(prefix=f"{db_config['id']}")

        result_id = int(re.search(r'Result ID:(\d+)', upload_response).group(1))

        # get result
        response = get_result(result_id=result_id)

        # save next config
        save_next_config(response, db_config, t=result_timestamp)

        print(f"The recommendation is: {response['recommendation']}")

        # change config
        change_conf(db_config, response['recommendation'])


def loop_task(server_conf: dict, iteration_num: int):
    # dump database if it's not done before.
    dump = dump_database(server_conf)
    reset_conf(server_conf, False)

    # restart database
    restart_succeeded = restart_database(server_conf)
    if not restart_succeeded:
        files = {'summary': b'{"error": "DB_RESTART_ERROR"}',
                    'knobs': b'{}',
                    'metrics_before': b'{}',
                    'metrics_after': b'{}'}
        if dconf.ENABLE_UDM:
            files['user_defined_metrics'] = b'{}'
        response = requests.post(dconf.WEBSITE_URL + '/new_result/', files=files,
                                    data={'upload_code': dconf.UPLOAD_CODE})
        response = get_result()
        result_timestamp = int(time.time())
        save_next_config(response, t=result_timestamp)
        change_conf(response['recommendation'])
        return

    # reload database periodically
    # if dconf.RELOAD_INTERVAL > 0:
    #     if iteration_num % dconf.RELOAD_INTERVAL == 0:
    #         is_ready_db(interval_sec=10)
    #         if iteration_num == 0 and dump is False:
    #             restore_database(server_conf)
    #         elif iteration_num > 0:
    #             restore_database(server_conf)
    LOG.info('Wait %s seconds after restarting database', dconf.RESTART_SLEEP_SEC)
    is_ready_db(interval_sec=10)

    LOG.info('The %s-th Loop Starts', iteration_num + 1)
    loop(server_conf, iteration_num % dconf.RELOAD_INTERVAL if dconf.RELOAD_INTERVAL > 0 else iteration_num)
    LOG.info('The %s-th Loop Ends', iteration_num + 1)


@task
def run_loops(max_iter=100):
    for i in range(int(max_iter)):
        db_processes = []
        for db_config in dconf.DB_SERVERS:
            p = Process(target=loop_task, args=(db_config, i))
            p.start()
            db_processes.append(p)

        for p in db_processes:
            print(f"Joining process {p}")
            p.join()
            print(f"Joined process {p}")

if __name__ == '__main__':
    run_loops(max_iter=sys.argv[1] if len(sys.argv) > 1 else 25)
