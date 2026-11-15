"""List the entry names of a remote zip without downloading it (HTTP range, zip64-aware)."""
import struct
import urllib.request

URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


size = 1839504686  # from HEAD request
tail = fetch(URL, size - 1048576, size - 1)

# zip64 EOCD locator (PK\x06\x07) then zip64 EOCD record (PK\x06\x06)
loc_sig, rec_sig = b"PK\x06\x07", b"PK\x06\x06"
zidx = tail.rfind(loc_sig)
assert zidx >= 0, "zip64 locator not found"
z64_eocd_off = struct.unpack("<IIQI", tail[zidx:zidx + 20])[2]
rec_off_in_tail = z64_eocd_off - (size - 1048576)
assert rec_off_in_tail >= 0 and tail[rec_off_in_tail:rec_off_in_tail + 4] == rec_sig, "zip64 EOCD not in tail"
(_, _, _, _, _, _, _, n_entries, cd_size, cd_offset) = struct.unpack("<IQHHIIQQQQ", tail[rec_off_in_tail:rec_off_in_tail + 56])
print(f"zip64 EOCD: entries={n_entries}, cd_size={cd_size:,}, cd_offset={cd_offset:,}")

cd = fetch(URL, cd_offset, cd_offset + cd_size - 1)
print("fetched central directory:", len(cd), "bytes")

names, pos, sig = [], 0, b"PK\x01\x02"
while pos < len(cd) and cd[pos:pos + 4] == sig:
    (nlen, elen, clen) = struct.unpack("<HHH", cd[pos + 28:pos + 34])
    names.append(cd[pos + 46:pos + 46 + nlen].decode("utf-8", "replace"))
    pos += 46 + nlen + elen + clen

print("parsed names:", len(names))
tops = sorted({n.split("/")[0] for n in names})
print("top-level entries:", tops)
top = tops[0]
print("csv entries:", [n for n in names if n.endswith(".csv")])
print("records100 file count:", sum(1 for n in names if "/records100/" in n and not n.endswith("/")))
print("records500 file count:", sum(1 for n in names if "/records500/" in n and not n.endswith("/")))
print("sample records100 entries:", [n for n in names if "/records100/" in n][:4])
print("first 6 entries:", names[:6])
