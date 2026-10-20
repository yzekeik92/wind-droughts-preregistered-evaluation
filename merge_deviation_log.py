#!/usr/bin/env python3
"""Merge Deviation Log cover + body into the final supplementary PDF."""
from pypdf import PdfReader, PdfWriter

A4_W, A4_H = 595.28, 841.89
COVER = '/home/z/my-project/scripts/devlog_cover.pdf'
BODY = '/home/z/my-project/scripts/devlog_body.pdf'
OUT = '/home/z/my-project/download/Deviation_Log.pdf'


def normalize(page):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    if abs(w - A4_W) > 0.1 or abs(h - A4_H) > 0.1:
        page.scale_to(A4_W, A4_H)
    return page


writer = PdfWriter()
writer.add_page(normalize(PdfReader(COVER).pages[0]))
for p in PdfReader(BODY).pages:
    writer.add_page(normalize(p))
writer.add_metadata({
    '/Title': 'Deviation Log: Pre-Specified Protocol Deviations and Post-Review Additions',
    '/Author': 'Z.ai',
    '/Creator': 'Z.ai',
    '/Subject': 'Supplementary material: deviations #11 to #19 with manuscript cross-references',
})
with open(OUT, 'wb') as f:
    writer.write(f)
print('merged:', OUT, '| pages:', len(writer.pages))
