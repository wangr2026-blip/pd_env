"""Structural diagnostics use synthetic containers; no real data required."""
from pathlib import Path
import struct
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_audio_failures import mp4_atoms,classify


def box(kind,payload=b''):
    return struct.pack('>I4s',len(payload)+8,kind)+payload


def test_header_only_has_no_recoverable_audio(tmp_path):
    p=tmp_path/'header.m4a';p.write_bytes(box(b'ftyp',b'\0'*20))
    atoms=mp4_atoms(p)
    assert atoms==[{'type':'ftyp','offset':0,'size':28,'payload_bytes':20,'complete':True}]
    assert classify(atoms,{'samples':0},{'samples':0})=='missing_metadata_and_audio_payload'


def test_payload_size_excludes_box_header(tmp_path):
    p=tmp_path/'bad.m4a';p.write_bytes(box(b'ftyp')+box(b'moov')+box(b'mdat',b'\0'*8))
    atoms=mp4_atoms(p)
    assert atoms[-1]['payload_bytes']==8
    assert classify(atoms,{'samples':0},{'samples':0})=='undecodable_audio_payload'
    assert classify(atoms,{'samples':0},{'samples':32})=='tolerant_decoder_produced_samples_review_required'


def test_truncated_extended_and_to_end_boxes(tmp_path):
    p=tmp_path/'boxes.m4a'
    p.write_bytes(struct.pack('>I4sQ',1,b'mdat',24)+b'\0'*8)
    assert mp4_atoms(p)[0]['payload_bytes']==8
    p.write_bytes(struct.pack('>I4s',0,b'mdat')+b'\0'*3)
    assert mp4_atoms(p)[0]['payload_bytes']==3
    p.write_bytes(struct.pack('>I4s',100,b'moov')+b'\0')
    assert mp4_atoms(p)[0]['complete'] is False
