#!/usr/bin/env bash
# Parallel-range download (aria2c, 8 connections per file, 3 files at once) + lbzip2 extract of the DNS4 shards in /root/scripts/select.txt, then MUSAN from the EU mirror.
cd /workspace/data && mkdir -p tar
B=https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband
dl() { n=$(basename "$1"); f=tar/$n; aria2c -q -x8 -s8 -k10M --file-allocation=none -c -d tar -o "$n" "$B/$1" && tar -I lbzip2 -xf "$f" -C /workspace/data && rm -f "$f" && echo "OK $1" || echo "FAIL $1"; }
export -f dl; export B
xargs -P 3 -I{} bash -c 'dl {}' < /root/scripts/select.txt
( cd tar && timeout 1500 aria2c -q -x8 -s8 -k10M --file-allocation=none -c -o musan.tar.gz https://us.openslr.org/resources/17/musan.tar.gz https://openslr.magicdatatech.com/resources/17/musan.tar.gz https://openslr.elda.org/resources/17/musan.tar.gz && tar -xzf musan.tar.gz -C /workspace/data && rm -f musan.tar.gz && echo "OK musan" || echo "SKIP musan (mirrors too slow)" )
echo DOWNLOAD_DONE; du -sh /workspace/data/*; df -h /workspace | tail -1
