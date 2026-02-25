#!/usr/bin/env python3
"""Sync rlog files from a comma device to a local directory.

Replaces both the broken shell script (rlog_copy_from_device_then_zip1.sh)
and the fragmented download tool (ryanomatic/rlog_aggregation) with a single
cross-platform rsync/SFTP wrapper.

Usage:
  python -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ~/nnlc-data/
  python -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ~/nnlc-data/ --dry-run
"""

import argparse
import os
import shutil
import subprocess
import sys
import stat

DEFAULT_USER = "comma"
DEFAULT_DEVICE_PATH = "/data/media/0/realdata/"
RLOG_EXTENSIONS = (".zst", ".bz2")


def sync_rsync(user, host, device_path, output_dir, dry_run=False):
    """Sync rlogs using rsync (fast, incremental)."""
    src = f"{user}@{host}:{device_path}"
    cmd = [
        "rsync", "-avz", "--progress",
        "--include=*/",
        "--include=rlog.zst",
        "--include=rlog.bz2",
        "--exclude=*",
        src, output_dir,
    ]
    if dry_run:
        cmd.insert(1, "--dry-run")

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def sync_sftp(user, host, device_path, output_dir, dry_run=False):
    """Sync rlogs using paramiko SFTP (fallback when rsync unavailable)."""
    try:
        import paramiko
    except ImportError:
        print("ERROR: paramiko not installed. Install with: pip install paramiko")
        return False

    print(f"Connecting to {user}@{host} via SFTP...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Try SSH agent first, then key files
    try:
        client.connect(host, username=user)
    except paramiko.AuthenticationException:
        key_paths = [
            os.path.expanduser("~/.ssh/id_ed25519"),
            os.path.expanduser("~/.ssh/id_rsa"),
        ]
        connected = False
        for key_path in key_paths:
            if os.path.exists(key_path):
                try:
                    client.connect(host, username=user, key_filename=key_path)
                    connected = True
                    break
                except paramiko.AuthenticationException:
                    continue
        if not connected:
            print("ERROR: Could not authenticate. Add your SSH key to the device.")
            return False

    sftp = client.open_sftp()
    synced = 0
    skipped = 0

    try:
        entries = sftp.listdir_attr(device_path)
    except FileNotFoundError:
        print(f"ERROR: Remote path not found: {device_path}")
        sftp.close()
        client.close()
        return False

    for entry in sorted(entries, key=lambda e: e.filename):
        if not stat.S_ISDIR(entry.st_mode):
            continue

        route_dir = entry.filename
        remote_route = os.path.join(device_path, route_dir)

        try:
            files = sftp.listdir(remote_route)
        except Exception:
            continue

        for fname in files:
            if not fname.startswith("rlog"):
                continue
            if not any(fname.endswith(ext) for ext in RLOG_EXTENSIONS):
                # Also grab uncompressed rlog
                if fname != "rlog":
                    continue

            remote_file = os.path.join(remote_route, fname)
            local_dir = os.path.join(output_dir, route_dir)
            local_file = os.path.join(local_dir, fname)

            if os.path.exists(local_file):
                # Skip if local file exists and has same size
                try:
                    remote_stat = sftp.stat(remote_file)
                    local_size = os.path.getsize(local_file)
                    if local_size == remote_stat.st_size:
                        skipped += 1
                        continue
                except Exception:
                    pass

            if dry_run:
                print(f"  [dry-run] Would download: {remote_file}")
                synced += 1
                continue

            os.makedirs(local_dir, exist_ok=True)
            print(f"  Downloading: {remote_file} -> {local_file}")
            sftp.get(remote_file, local_file)
            synced += 1

    sftp.close()
    client.close()

    action = "Would sync" if dry_run else "Synced"
    print(f"\n{action} {synced} files, skipped {skipped} (already present)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync rlog files from a comma device to a local directory.",
    )
    parser.add_argument("-d", "--device", required=True,
                        help="Device IP address (e.g. 192.168.1.161)")
    parser.add_argument("-o", "--output", required=True,
                        help="Local output directory for rlogs")
    parser.add_argument("-u", "--user", default=DEFAULT_USER,
                        help=f"SSH username (default: {DEFAULT_USER})")
    parser.add_argument("-p", "--path", default=DEFAULT_DEVICE_PATH,
                        help=f"Device rlog path (default: {DEFAULT_DEVICE_PATH})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be synced without downloading")
    parser.add_argument("--no-rsync", action="store_true",
                        help="Force SFTP mode even if rsync is available")
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # Ensure device path ends with /
    device_path = args.path if args.path.endswith("/") else args.path + "/"

    has_rsync = shutil.which("rsync") is not None and not args.no_rsync

    if has_rsync:
        print("Using rsync for fast incremental sync...")
        success = sync_rsync(args.user, args.device, device_path, output_dir, args.dry_run)
    else:
        print("rsync not available, using SFTP fallback...")
        success = sync_sftp(args.user, args.device, device_path, output_dir, args.dry_run)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
